#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "json.hpp" // For JSON settings file
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

#ifdef ENABLE_HAND_TRACKING
#include "hand_tracker.hpp"
#endif

// Use nlohmann::json for convenience
using json = nlohmann::json;

// Struct to hold command-line arguments
struct AppConfig {
    bool show_timestamp = false;
    std::string mode = "tracking";
    int width = 0;
    int height = 0;
    int fps = 0;
    double downscale_factor = 0.5; // Default to 50% downscaling for performance
    bool high_fps_preferred = false;
    bool track_hands = false;
};

// Camera configuration presets
struct CameraMode {
    int width, height, fps;
};

// Minimum contour area to filter out noise
const double MIN_CONTOUR_AREA = 100.0;
// Maximum depth value to consider (in meters)
const float MAX_DEPTH = 3.0f;
// Configuration file name
const std::string SETTINGS_FILE = "ball_settings.json";
// Distance threshold for merging nearby detections (pixels)
const double MERGE_DISTANCE_THRESHOLD = 80.0;

// Struct to hold HSV color range
struct ColorRange {
    std::string name;
    cv::Scalar min_hsv;
    cv::Scalar max_hsv;
    cv::Scalar min_hsv2; // For colors that might wrap around HSV
    cv::Scalar max_hsv2; // For colors that might wrap around HSV
    
    ColorRange(const std::string& n, const cv::Scalar& min, const cv::Scalar& max, 
               const cv::Scalar& min2 = cv::Scalar(-1, -1, -1), const cv::Scalar& max2 = cv::Scalar(-1, -1, -1))
        : name(n), min_hsv(min), max_hsv(max), min_hsv2(min2), max_hsv2(max2) {}
};

// Struct for final ball detection data
struct BallDetection {
    std::string name;
    cv::Point2f center;
    float world_x, world_y, world_z;
    float confidence; // For merging decisions
    bool is_held = false;
};

// Global variables for calibration mode
struct CalibrationState {
    cv::Mat current_frame;
    cv::Mat hsv_frame;
    std::vector<ColorRange>* colors;
    int selected_color_index = 0;
    bool mouse_clicked = false;
    cv::Point click_point;
    std::string window_name;
};

CalibrationState cal_state;

// Function to calculate distance between two points
double calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

float calculate_3d_distance(float x1, float y1, float z1, float x2, float y2, float z2) {
    return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2) + std::pow(z1 - z2, 2));
}

// Function to merge nearby detections of the same color
std::vector<cv::Point2f> mergeNearbyDetections(const std::vector<cv::Point2f>& centers) {
    if (centers.empty()) return centers;
    
    std::vector<cv::Point2f> merged_centers;
    std::vector<bool> used(centers.size(), false);
    
    for (size_t i = 0; i < centers.size(); ++i) {
        if (used[i]) continue;
        
        // Start a new cluster with this center
        std::vector<cv::Point2f> cluster;
        cluster.push_back(centers[i]);
        used[i] = true;
        
        // Find all nearby centers to merge
        for (size_t j = i + 1; j < centers.size(); ++j) {
            if (used[j]) continue;
            
            // Check if this center is close to any center in the current cluster
            bool should_merge = false;
            for (const auto& cluster_center : cluster) {
                if (calculateDistance(centers[j], cluster_center) < MERGE_DISTANCE_THRESHOLD) {
                    should_merge = true;
                    break;
                }
            }
            
            if (should_merge) {
                cluster.push_back(centers[j]);
                used[j] = true;
            }
        }
        
        // Calculate the centroid of the cluster
        cv::Point2f centroid(0, 0);
        for (const auto& point : cluster) {
            centroid.x += point.x;
            centroid.y += point.y;
        }
        centroid.x /= cluster.size();
        centroid.y /= cluster.size();
        
        merged_centers.push_back(centroid);
    }
    
    return merged_centers;
}

// Function to save settings to a JSON file
void saveSettings(const std::vector<ColorRange>& colors) {
    json j;
    for (const auto& color : colors) {
        j[color.name]["min_hsv"] = {color.min_hsv[0], color.min_hsv[1], color.min_hsv[2]};
        j[color.name]["max_hsv"] = {color.max_hsv[0], color.max_hsv[1], color.max_hsv[2]};
        if (color.min_hsv2[0] >= 0) { // Only save second range if it's valid
            j[color.name]["min_hsv2"] = {color.min_hsv2[0], color.min_hsv2[1], color.min_hsv2[2]};
            j[color.name]["max_hsv2"] = {color.max_hsv2[0], color.max_hsv2[1], color.max_hsv2[2]};
        }
    }
    std::ofstream file(SETTINGS_FILE);
    file << j.dump(4); // pretty print with 4 spaces
    std::cerr << "Settings saved to " << SETTINGS_FILE << std::endl;
}

// Function to load settings from a JSON file
bool loadSettings(std::vector<ColorRange>& colors) {
    std::ifstream file(SETTINGS_FILE);
    if (!file.is_open()) {
        std::cerr << "Warning: Settings file not found. Using default values." << std::endl;
        return false;
    }
    
    try {
        json j;
        file >> j;
        for (auto& color : colors) {
            if (j.contains(color.name)) {
                auto& color_data = j[color.name];
                color.min_hsv = cv::Scalar(color_data["min_hsv"][0], color_data["min_hsv"][1], color_data["min_hsv"][2]);
                color.max_hsv = cv::Scalar(color_data["max_hsv"][0], color_data["max_hsv"][1], color_data["max_hsv"][2]);
                if (color_data.contains("min_hsv2")) {
                    color.min_hsv2 = cv::Scalar(color_data["min_hsv2"][0], color_data["min_hsv2"][1], color_data["min_hsv2"][2]);
                    color.max_hsv2 = cv::Scalar(color_data["max_hsv2"][0], color_data["max_hsv2"][1], color_data["max_hsv2"][2]);
                }
            }
        }
        std::cerr << "Settings loaded from " << SETTINGS_FILE << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading settings: " << e.what() << ". Using default values." << std::endl;
        return false;
    }
}

// Mouse callback function for calibration
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        cal_state.mouse_clicked = true;
        cal_state.click_point = cv::Point(x, y);
        
        if (!cal_state.hsv_frame.empty() && cal_state.colors) {
            // Sample a 5x5 area around the click point
            int sample_size = 5;
            int start_x = std::max(0, x - sample_size/2);
            int start_y = std::max(0, y - sample_size/2);
            int end_x = std::min(cal_state.hsv_frame.cols - 1, x + sample_size/2);
            int end_y = std::min(cal_state.hsv_frame.rows - 1, y + sample_size/2);
            
            cv::Rect sample_rect(start_x, start_y, end_x - start_x, end_y - start_y);
            cv::Mat sample_area = cal_state.hsv_frame(sample_rect);
            
            // Calculate mean and standard deviation of HSV values in the sample area
            cv::Scalar mean, stddev;
            cv::meanStdDev(sample_area, mean, stddev);
            
            // Set HSV range based on the sampled values with reduced tolerance for better separation
            int h_tolerance = 8;  // Reduced from 15 to prevent color overlap
            int s_tolerance = 40; // Reduced from 50
            int v_tolerance = 40; // Reduced from 50
            
            auto& selected_color = (*cal_state.colors)[cal_state.selected_color_index];
            
            // Handle potential HSV wrap-around for colors near red/magenta boundary
            int h_mean = (int)mean[0];
            if ((selected_color.name == "pink" && (h_mean <= 10 || h_mean >= 170)) ||
                (selected_color.name == "orange" && h_mean <= 15)) {
                // Handle wrap-around
                if (h_mean <= 15) {
                    selected_color.min_hsv = cv::Scalar(
                        std::max(0, h_mean - h_tolerance),
                        std::max(0, (int)mean[1] - s_tolerance),
                        std::max(0, (int)mean[2] - v_tolerance)
                    );
                    selected_color.max_hsv = cv::Scalar(
                        std::min(15, h_mean + h_tolerance),
                        std::min(255, (int)mean[1] + s_tolerance),
                        std::min(255, (int)mean[2] + v_tolerance)
                    );
                    selected_color.min_hsv2 = cv::Scalar(
                        std::max(165, 180 - h_tolerance),
                        std::max(0, (int)mean[1] - s_tolerance),
                        std::max(0, (int)mean[2] - v_tolerance)
                    );
                    selected_color.max_hsv2 = cv::Scalar(
                        180,
                        std::min(255, (int)mean[1] + s_tolerance),
                        std::min(255, (int)mean[2] + v_tolerance)
                    );
                } else {
                    selected_color.min_hsv = cv::Scalar(
                        std::max(165, h_mean - h_tolerance),
                        std::max(0, (int)mean[1] - s_tolerance),
                        std::max(0, (int)mean[2] - v_tolerance)
                    );
                    selected_color.max_hsv = cv::Scalar(
                        180,
                        std::min(255, (int)mean[1] + s_tolerance),
                        std::min(255, (int)mean[2] + v_tolerance)
                    );
                    selected_color.min_hsv2 = cv::Scalar(
                        0,
                        std::max(0, (int)mean[1] - s_tolerance),
                        std::max(0, (int)mean[2] - v_tolerance)
                    );
                    selected_color.max_hsv2 = cv::Scalar(
                        std::min(15, h_tolerance),
                        std::min(255, (int)mean[1] + s_tolerance),
                        std::min(255, (int)mean[2] + v_tolerance)
                    );
                }
            } else {
                // Normal single range
                selected_color.min_hsv = cv::Scalar(
                    std::max(0, h_mean - h_tolerance),
                    std::max(0, (int)mean[1] - s_tolerance),
                    std::max(0, (int)mean[2] - v_tolerance)
                );
                selected_color.max_hsv = cv::Scalar(
                    std::min(180, h_mean + h_tolerance),
                    std::min(255, (int)mean[1] + s_tolerance),
                    std::min(255, (int)mean[2] + v_tolerance)
                );
                selected_color.min_hsv2 = cv::Scalar(-1, -1, -1); // Disable second range
                selected_color.max_hsv2 = cv::Scalar(-1, -1, -1);
            }
            
            std::cerr << "Calibrated " << selected_color.name << " color from click at (" << x << "," << y << ")" << std::endl;
            std::cerr << "HSV values - H:" << (int)mean[0] << " S:" << (int)mean[1] << " V:" << (int)mean[2] << std::endl;
        }
    }
}

// Unified function to detect balls for a given color range with merging and downscaling
void detectBalls(const cv::Mat& hsv_frame, const ColorRange& color, std::vector<cv::Point2f>& centers, double downscale_factor = 1.0) {
    if (downscale_factor == 1.0) {
        // Original processing path
        cv::Mat mask;
        cv::inRange(hsv_frame, color.min_hsv, color.max_hsv, mask);
        if (color.min_hsv2[0] >= 0) {
            cv::Mat mask2;
            cv::inRange(hsv_frame, color.min_hsv2, color.max_hsv2, mask2);
            cv::bitwise_or(mask, mask2, mask);
        }

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point2f> initial_centers;
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) > MIN_CONTOUR_AREA) {
                cv::Moments m = cv::moments(contour);
                if (m.m00 > 0) {
                    initial_centers.push_back(cv::Point2f(m.m10 / m.m00, m.m01 / m.m00));
                }
            }
        }
        centers = mergeNearbyDetections(initial_centers);

    } else {
        // Optimized path with downscaling
        cv::Mat resized_hsv;
        cv::resize(hsv_frame, resized_hsv, cv::Size(), downscale_factor, downscale_factor, cv::INTER_LINEAR);

        cv::Mat mask;
        cv::inRange(resized_hsv, color.min_hsv, color.max_hsv, mask);
        if (color.min_hsv2[0] >= 0) {
            cv::Mat mask2;
            cv::inRange(resized_hsv, color.min_hsv2, color.max_hsv2, mask2);
            cv::bitwise_or(mask, mask2, mask);
        }

        // Use smaller kernel for smaller image
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        std::vector<cv::Point2f> initial_centers;
        double scaled_min_area = MIN_CONTOUR_AREA * downscale_factor * downscale_factor;
        for (const auto& contour : contours) {
            if (cv::contourArea(contour) > scaled_min_area) {
                cv::Moments m = cv::moments(contour);
                if (m.m00 > 0) {
                    initial_centers.push_back(cv::Point2f(m.m10 / m.m00, m.m01 / m.m00));
                }
            }
        }
        
        // Scale centers back to original image size
        for (auto& center : initial_centers) {
            center.x /= downscale_factor;
            center.y /= downscale_factor;
        }
        centers = mergeNearbyDetections(initial_centers);
    }
}

// Interactive calibration mode
void runCalibration(std::vector<ColorRange>& colors) {
    std::cerr << "Starting calibration mode..." << std::endl;
    
    // Initialize RealSense pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    
    // Try different resolutions for calibration
    try {
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
        pipe.start(cfg);
        std::cerr << "Calibration started at 1280x720 @ 30 FPS" << std::endl;
    } catch (const rs2::error& e) {
        std::cerr << "Failed to start calibration: " << e.what() << std::endl;
        return;
    }

    const std::string window_name = "Ball Tracker Calibration - Click on balls to calibrate";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    
    // Set up calibration state
    cal_state.colors = &colors;
    cal_state.window_name = window_name;
    
    // Set mouse callback
    cv::setMouseCallback(window_name, onMouse, nullptr);
    
    std::cerr << "\n=== CALIBRATION INSTRUCTIONS ===" << std::endl;
    std::cerr << "1. Click on a ball in the camera feed to calibrate its color" << std::endl;
    std::cerr << "2. Use number keys to select color to calibrate:" << std::endl;
    std::cerr << "   '1' = Pink, '2' = Orange, '3' = Green, '4' = Yellow" << std::endl;
    std::cerr << "3. Press 's' to save settings" << std::endl;
    std::cerr << "4. Press 'r' to reset to defaults" << std::endl;
    std::cerr << "5. Press 'q' or ESC to quit" << std::endl;
    std::cerr << "\nCurrently calibrating: " << colors[cal_state.selected_color_index].name << std::endl;

    while (true) {
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 100)) {
            continue;
        }
        
        rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) continue;
        
        cv::Mat color_image(cv::Size(1280, 720), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat hsv_image, display_image;
        cv::cvtColor(color_image, hsv_image, cv::COLOR_BGR2HSV);
        color_image.copyTo(display_image);
        
        // Update global state for mouse callback
        cal_state.current_frame = color_image;
        cal_state.hsv_frame = hsv_image;
        
        // Generate visualization with detected balls
        for (size_t i = 0; i < colors.size(); ++i) {
            const auto& color = colors[i];
            std::vector<cv::Point2f> centers;
            detectBalls(hsv_image, color, centers);
            
            // Draw detected balls
            cv::Scalar draw_color;
            if (color.name == "pink") draw_color = cv::Scalar(147, 20, 255);      // Pink in BGR
            else if (color.name == "orange") draw_color = cv::Scalar(0, 165, 255); // Orange in BGR
            else if (color.name == "green") draw_color = cv::Scalar(0, 255, 0);    // Green in BGR
            else if (color.name == "yellow") draw_color = cv::Scalar(0, 255, 255); // Yellow in BGR
            
            // Highlight selected color
            int thickness = (i == cal_state.selected_color_index) ? 5 : 3;
            
            for (const auto& center : centers) {
                cv::circle(display_image, center, 15, draw_color, thickness);
                cv::putText(display_image, color.name, cv::Point(center.x - 20, center.y - 20),
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2);
            }
        }
        
        // Draw crosshair at last click point
        if (cal_state.mouse_clicked) {
            cv::drawMarker(display_image, cal_state.click_point, cv::Scalar(255, 255, 255), 
                          cv::MARKER_CROSS, 20, 3);
        }
        
        // Add instructions and current color info to the image
        std::string current_color = colors[cal_state.selected_color_index].name;
        cv::putText(display_image, "Currently calibrating: " + current_color + " (Press 1/2/3/4 to change)", 
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(display_image, "Click on a " + current_color + " ball to calibrate", 
                   cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        cv::putText(display_image, "Press 's' to save, 'r' to reset, 'q' to quit", 
                   cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        // Show HSV ranges for current color
        const auto& current = colors[cal_state.selected_color_index];
        std::string hsv_info = current.name + " HSV: [" + 
                              std::to_string((int)current.min_hsv[0]) + "-" + std::to_string((int)current.max_hsv[0]) + ", " +
                              std::to_string((int)current.min_hsv[1]) + "-" + std::to_string((int)current.max_hsv[1]) + ", " +
                              std::to_string((int)current.min_hsv[2]) + "-" + std::to_string((int)current.max_hsv[2]) + "]";
        cv::putText(display_image, hsv_info, cv::Point(10, display_image.rows - 20), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        
        cv::imshow(window_name, display_image);

        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 27) { // q or ESC
            break;
        }
        if (key == 's') {
            saveSettings(colors);
        }
        if (key == 'r') {
            // Reset to default values optimized for separation
            colors[0] = ColorRange("pink", cv::Scalar(150, 150, 90), cv::Scalar(170, 255, 255));
            colors[1] = ColorRange("orange", cv::Scalar(5, 150, 120), cv::Scalar(15, 255, 255));
            colors[2] = ColorRange("green", cv::Scalar(45, 120, 70), cv::Scalar(75, 255, 255));
            colors[3] = ColorRange("yellow", cv::Scalar(25, 120, 100), cv::Scalar(35, 255, 255));
            std::cerr << "Reset to default values" << std::endl;
        }
        if (key == '1') {
            cal_state.selected_color_index = 0;
            std::cerr << "Now calibrating: " << colors[0].name << std::endl;
        }
        if (key == '2') {
            cal_state.selected_color_index = 1;
            std::cerr << "Now calibrating: " << colors[1].name << std::endl;
        }
        if (key == '3') {
            cal_state.selected_color_index = 2;
            std::cerr << "Now calibrating: " << colors[2].name << std::endl;
        }
        if (key == '4') {
            cal_state.selected_color_index = 3;
            std::cerr << "Now calibrating: " << colors[3].name << std::endl;
        }
    }
    
    cv::destroyAllWindows();
    std::cerr << "Calibration mode ended" << std::endl;
}

// Function to encode image as base64
std::string encodeImageToBase64(const cv::Mat& image) {
    std::vector<uchar> buffer;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80}; // Compress to reduce size
    cv::imencode(".jpg", image, buffer, params);
    
    // Convert to base64
    std::string encoded;
    const char* chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int val = 0, valb = -6;
    for (uchar c : buffer) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            encoded.push_back(chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) encoded.push_back(chars[((val << 8) >> (valb + 8)) & 0x3F]);
    while (encoded.size() % 4) encoded.push_back('=');
    return encoded;
}

// Helper function to get an averaged depth value from a patch of pixels
float get_averaged_depth(const rs2::depth_frame& depth_frame, int x, int y, int patch_size) {
    float total_depth = 0.0f;
    int valid_pixel_count = 0;
    
    // Define the area to sample
    int start_x = std::max(0, x - patch_size / 2);
    int end_x = std::min(depth_frame.get_width() - 1, x + patch_size / 2);
    int start_y = std::max(0, y - patch_size / 2);
    int end_y = std::min(depth_frame.get_height() - 1, y + patch_size / 2);

    for (int cy = start_y; cy <= end_y; ++cy) {
        for (int cx = start_x; cx <= end_x; ++cx) {
            float depth = depth_frame.get_distance(cx, cy);
            // Only include valid depth readings (ignore 0 values)
            if (depth > 0.0f) {
                total_depth += depth;
                valid_pixel_count++;
            }
        }
    }

    if (valid_pixel_count == 0) {
        return 0.0f; // Return 0 if no valid depth was found
    }

    return total_depth / valid_pixel_count;
}

// Main tracking logic
void runTracking(std::vector<ColorRange>& colors, const AppConfig& config
#ifdef ENABLE_HAND_TRACKING
, std::unique_ptr<HandTracker>& hand_tracker
#endif
) {
    std::cerr << "Starting tracking mode..." << std::endl;
    
    rs2::pipeline pipe;
    rs2::config cfg;
    int width = 0, height = 0;

    // Define a list of prioritized camera modes to try
    std::vector<CameraMode> modes_to_try;
    if (config.high_fps_preferred) {
        modes_to_try = {{848, 480, 90}, {640, 480, 60}, {1280, 720, 30}};
    } else {
        modes_to_try = {{1280, 720, 90}, {1280, 720, 60}, {1280, 720, 30}, {848, 480, 90}, {640, 480, 60}};
    }

    bool started = false;
    for (const auto& mode : modes_to_try) {
        try {
            cfg.disable_all_streams();
            cfg.enable_stream(RS2_STREAM_COLOR, mode.width, mode.height, RS2_FORMAT_BGR8, mode.fps);
            cfg.enable_stream(RS2_STREAM_DEPTH, mode.width, mode.height, RS2_FORMAT_Z16, mode.fps);
            pipe.start(cfg);
            width = mode.width;
            height = mode.height;
            std::cerr << "Successfully started camera at " << width << "x" << height << " @ " << mode.fps << " FPS" << std::endl;
            started = true;
            break;
        } catch (const rs2::error& e) {
            std::cerr << "Warning: Could not start " << mode.width << "x" << mode.height << " @ " << mode.fps << " FPS. Trying next mode..." << std::endl;
        }
    }

    if (!started) {
        throw std::runtime_error("Failed to start RealSense camera with any available mode.");
    }

    // Get camera intrinsics and create alignment object
    auto profile = pipe.get_active_profile();
    auto intrinsics = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    rs2::align align_to_color(RS2_STREAM_COLOR);

    // 1. Create post-processing filter objects
    rs2::spatial_filter spat_filter;
    rs2::temporal_filter temp_filter;

    std::cerr << "RealSense post-processing filters enabled." << std::endl;

    std::cerr << "Ball tracker initialized. Press Ctrl+C to stop." << std::endl;
    
    while (true) {
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 100)) {
            continue;
        }
        
        auto aligned_frames = align_to_color.process(frames);
        auto color_frame = aligned_frames.get_color_frame();
        rs2::depth_frame depth_frame = aligned_frames.get_depth_frame(); // Get non-const depth frame
        
        if (!color_frame || !depth_frame) continue;

        // 2. Apply the filters to the depth frame
        // The order is important: spatial first, then temporal.
        depth_frame = spat_filter.process(depth_frame);
        depth_frame = temp_filter.process(depth_frame);

        // Convert frames to OpenCV matrices
        cv::Mat color_image(cv::Size(width, height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat hsv_image;
        cv::cvtColor(color_image, hsv_image, cv::COLOR_BGR2HSV);

        std::vector<BallDetection> all_detections;
        
        // Detect balls for each color
        for (const auto& color : colors) {
            std::vector<cv::Point2f> centers;
            detectBalls(hsv_image, color, centers, config.downscale_factor);

            for (const auto& center : centers) {
                int x = static_cast<int>(center.x);
                int y = static_cast<int>(center.y);
                
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    // 3. NEW, MORE ROBUST WAY:
                    const int patch_size = 5; // Use a 5x5 pixel area
                    float depth = get_averaged_depth(depth_frame, x, y, patch_size);

                    if (depth > 0 && depth < MAX_DEPTH) {
                        // Deproject 2D pixel to 3D point
                        float pixel[2] = {center.x, center.y};
                        float point[3];
                        rs2_deproject_pixel_to_point(point, &intrinsics, pixel, depth);

                        all_detections.push_back({color.name, center, point[0], point[1], point[2], 1.0f});
                    }
                }
            }
        }

#ifdef ENABLE_HAND_TRACKING
        // Hand tracking
        if (config.track_hands && hand_tracker) {
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            
            auto hands = hand_tracker->DetectHands(color_image, timestamp);
            
            for (size_t i = 0; i < hands.size(); ++i) {
                for (size_t j = 0; j < hands[i].normalized_landmarks.size(); ++j) {
                    const auto& landmark = hands[i].normalized_landmarks[j];
                    int lx = static_cast<int>(landmark.x * width);
                    int ly = static_cast<int>(landmark.y * height);

                    if (lx >= 0 && lx < width && ly >= 0 && ly < height) {
                        float depth = get_averaged_depth(depth_frame, lx, ly, 3); // 3x3 patch
                        if (depth > 0 && depth < MAX_DEPTH) {
                            float pixel[2] = { (float)lx, (float)ly };
                            float point[3];
                            rs2_deproject_pixel_to_point(point, &intrinsics, pixel, depth);
                            
                            // Add hand landmark data to detections
                            all_detections.push_back({"hand" + std::to_string(i) + "_landmark" + std::to_string(j), cv::Point2f(lx, ly), point[0], point[1], point[2], 1.0f});
                        }
                    }
                }
            }
        }
#endif

        // Output results in the specified format
        if (!all_detections.empty()) {
            std::string output_line;
            if (config.show_timestamp) {
                auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()
                ).count();
                output_line += std::to_string(timestamp) + "|";
            }
            for (size_t i = 0; i < all_detections.size(); ++i) {
                const auto& ball = all_detections[i];
                output_line += ball.name + "," +
                               std::to_string(ball.world_x) + "," +
                               std::to_string(ball.world_y) + "," +
                               std::to_string(ball.world_z);
                if (i < all_detections.size() - 1) {
                    output_line += ";";
                }
            }
            std::cout << output_line << std::endl;
            std::cout.flush();
        }
        
        // Release matrices to prevent memory leaks
        color_image.release();
        hsv_image.release();
    }
}

// Streaming mode with video frames and tracking data
void runStreaming(std::vector<ColorRange>& colors, const AppConfig& config
#ifdef ENABLE_HAND_TRACKING
, std::unique_ptr<HandTracker>& hand_tracker
#endif
) {
    std::cerr << "Starting streaming mode..." << std::endl;
    
    rs2::pipeline pipe;
    rs2::config cfg;
    int width = 0, height = 0;
    bool started = false;

    // 1. Try user-specified settings first
    if (config.width > 0 && config.height > 0 && config.fps > 0) {
        try {
            cfg.enable_stream(RS2_STREAM_COLOR, config.width, config.height, RS2_FORMAT_BGR8, config.fps);
            cfg.enable_stream(RS2_STREAM_DEPTH, config.width, config.height, RS2_FORMAT_Z16, config.fps);
            pipe.start(cfg);
            width = config.width;
            height = config.height;
            std::cerr << "Successfully started camera with user settings: " << width << "x" << height << " @ " << config.fps << " FPS" << std::endl;
            started = true;
        } catch (const rs2::error& e) {
            std::cerr << "Warning: Could not start with user-specified settings. " << e.what() << ". Falling back to default modes." << std::endl;
        }
    }

    // 2. If user settings fail or are not provided, use intelligent fallback
    if (!started) {
        std::vector<CameraMode> modes_to_try;
        if (config.high_fps_preferred) {
            modes_to_try = {{848, 480, 90}, {640, 480, 60}, {1280, 720, 30}};
        } else {
            modes_to_try = {{1280, 720, 90}, {1280, 720, 60}, {1280, 720, 30}, {848, 480, 90}, {640, 480, 60}};
        }

        for (const auto& mode : modes_to_try) {
            try {
                cfg.disable_all_streams();
                cfg.enable_stream(RS2_STREAM_COLOR, mode.width, mode.height, RS2_FORMAT_BGR8, mode.fps);
                cfg.enable_stream(RS2_STREAM_DEPTH, mode.width, mode.height, RS2_FORMAT_Z16, mode.fps);
                pipe.start(cfg);
                width = mode.width;
                height = mode.height;
                std::cerr << "Successfully started camera at " << width << "x" << height << " @ " << mode.fps << " FPS" << std::endl;
                started = true;
                break;
            } catch (const rs2::error& e) {
                std::cerr << "Info: Could not start " << mode.width << "x" << mode.height << " @ " << mode.fps << ". Trying next mode." << std::endl;
            }
        }
    }

    if (!started) {
        throw std::runtime_error("Failed to start RealSense camera with any available mode.");
    }

    // Get camera intrinsics and create alignment object
    auto profile = pipe.get_active_profile();
    auto intrinsics = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    rs2::align align_to_color(RS2_STREAM_COLOR);

    // 1. Create post-processing filter objects
    rs2::spatial_filter spat_filter;
    rs2::temporal_filter temp_filter;

    std::cerr << "RealSense post-processing filters enabled." << std::endl;

    std::cerr << "Ball tracker streaming initialized. Press Ctrl+C to stop." << std::endl;
    
    while (true) {
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 100)) {
            continue;
        }
        
        auto aligned_frames = align_to_color.process(frames);
        auto color_frame = aligned_frames.get_color_frame();
        rs2::depth_frame depth_frame = aligned_frames.get_depth_frame(); // Get non-const depth frame
        
        if (!color_frame || !depth_frame) continue;

        // 2. Apply the filters to the depth frame
        // The order is important: spatial first, then temporal.
        depth_frame = spat_filter.process(depth_frame);
        depth_frame = temp_filter.process(depth_frame);

        // Convert frames to OpenCV matrices
        cv::Mat color_image(cv::Size(width, height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat hsv_image;
        cv::cvtColor(color_image, hsv_image, cv::COLOR_BGR2HSV);

        // Create display image with ball overlays
        cv::Mat display_image = color_image.clone();
        std::vector<BallDetection> all_detections;
        
        // Detect balls for each color
        for (const auto& color : colors) {
            std::vector<cv::Point2f> centers;
            detectBalls(hsv_image, color, centers, config.downscale_factor);

            // Draw color for visualization
            cv::Scalar draw_color;
            if (color.name == "pink") draw_color = cv::Scalar(147, 20, 255);      // Pink in BGR
            else if (color.name == "orange") draw_color = cv::Scalar(0, 165, 255); // Orange in BGR
            else if (color.name == "green") draw_color = cv::Scalar(0, 255, 0);    // Green in BGR
            else if (color.name == "yellow") draw_color = cv::Scalar(0, 255, 255); // Yellow in BGR

            for (const auto& center : centers) {
                int x = static_cast<int>(center.x);
                int y = static_cast<int>(center.y);
                
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    // 3. NEW, MORE ROBUST WAY:
                    const int patch_size = 5; // Use a 5x5 pixel area
                    float depth = get_averaged_depth(depth_frame, x, y, patch_size);
                    if (depth > 0 && depth < MAX_DEPTH) {
                        // Deproject 2D pixel to 3D point
                        float pixel[2] = {center.x, center.y};
                        float point[3];
                        rs2_deproject_pixel_to_point(point, &intrinsics, pixel, depth);

                        all_detections.push_back({color.name, center, point[0], point[1], point[2], 1.0f});
                        
                        // Draw ball on display image
                        cv::circle(display_image, center, 15, draw_color, 3);
                        cv::putText(display_image, color.name, cv::Point(center.x - 20, center.y - 20),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2);
                        
                        // Add 3D coordinates text
                        std::string coord_text = "(" + std::to_string(point[0]).substr(0, 4) + "," +
                                               std::to_string(point[1]).substr(0, 4) + "," +
                                               std::to_string(point[2]).substr(0, 4) + ")";
                        cv::putText(display_image, coord_text, cv::Point(center.x - 30, center.y + 30),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1);
                    }
                }
            }
        }

#ifdef ENABLE_HAND_TRACKING
        // Hand tracking visualization
        if (config.track_hands && hand_tracker) {
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();

            auto hands = hand_tracker->DetectHands(color_image, timestamp);

            for (size_t i = 0; i < hands.size(); ++i) {
                for (size_t j = 0; j < hands[i].normalized_landmarks.size(); ++j) {
                    const auto& landmark = hands[i].normalized_landmarks[j];
                    int lx = static_cast<int>(landmark.x * width);
                    int ly = static_cast<int>(landmark.y * height);

                    if (lx >= 0 && lx < width && ly >= 0 && ly < height) {
                        float depth = get_averaged_depth(depth_frame, lx, ly, 3);
                        if (depth > 0 && depth < MAX_DEPTH) {
                            float pixel[2] = { (float)lx, (float)ly };
                            float point[3];
                            rs2_deproject_pixel_to_point(point, &intrinsics, pixel, depth);
                            
                            all_detections.push_back({"hand" + std::to_string(i) + "_landmark" + std::to_string(j), cv::Point2f(lx, ly), point[0], point[1], point[2], 1.0f});

                            // Draw landmark on display image
                            cv::circle(display_image, cv::Point(lx, ly), 5, cv::Scalar(0, 255, 255), -1);
                        }
                    }
                }
            }
        }
#endif

        // Output tracking data
        std::string tracking_data;
        if (config.show_timestamp) {
            auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()
            ).count();
            tracking_data += std::to_string(timestamp) + "|";
        }
        if (!all_detections.empty()) {
            for (size_t i = 0; i < all_detections.size(); ++i) {
                const auto& ball = all_detections[i];
                tracking_data += ball.name + "," +
                               std::to_string(ball.world_x) + "," +
                               std::to_string(ball.world_y) + "," +
                               std::to_string(ball.world_z);
                if (i < all_detections.size() - 1) {
                    tracking_data += ";";
                }
            }
        }

        // Encode image to base64
        std::string encoded_image = encodeImageToBase64(display_image);
        
        // Output in format: FRAME:<base64_image>|TRACK:<tracking_data>
        std::cout << "FRAME:" << encoded_image << "|TRACK:" << tracking_data << std::endl;
        std::cout.flush();
        
        // Release matrices to prevent memory leaks
        color_image.release();
        hsv_image.release();
        display_image.release();
    }
}

// Function to parse command-line arguments
AppConfig parseArguments(int argc, char* argv[]) {
    AppConfig config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--timestamp" || arg == "-t") {
            config.show_timestamp = true;
        } else if (arg == "--high-fps" || arg == "-r") {
            config.high_fps_preferred = true;
        } else if (arg == "--width" && i + 1 < argc) {
            config.width = std::stoi(argv[++i]);
        } else if (arg == "--height" && i + 1 < argc) {
            config.height = std::stoi(argv[++i]);
        } else if (arg == "--fps" && i + 1 < argc) {
            config.fps = std::stoi(argv[++i]);
        } else if (arg == "--downscale" && i + 1 < argc) {
            config.downscale_factor = std::stod(argv[++i]);
        } else if (arg == "--track-hands") {
            config.track_hands = true;
        } else if (arg == "calibrate" || arg == "stream" || arg == "tracking") {
            config.mode = arg;
        } else {
            std::cerr << "Warning: Unknown argument '" << arg << "' ignored." << std::endl;
        }
    }
    return config;
}

int main(int argc, char* argv[]) {
    try {
        // Parse command-line arguments
        AppConfig config = parseArguments(argc, argv);

        // Default color values
        std::vector<ColorRange> colors = {
            ColorRange("pink", cv::Scalar(150, 150, 90), cv::Scalar(170, 255, 255)),
            ColorRange("orange", cv::Scalar(5, 150, 120), cv::Scalar(15, 255, 255)),
            ColorRange("green", cv::Scalar(45, 120, 70), cv::Scalar(75, 255, 255)),
            ColorRange("yellow", cv::Scalar(25, 120, 100), cv::Scalar(35, 255, 255))
        };
        
        // Load settings from file
        loadSettings(colors);

#ifdef ENABLE_HAND_TRACKING
    std::unique_ptr<HandTracker> hand_tracker;
    if (config.track_hands) {
        try {
            // IMPORTANT: User must provide the correct path to the downloaded .task model
            std::string model_path = "/home/twain/Projects/mediapipe/hand_landmarker.task";
            hand_tracker = std::make_unique<HandTracker>(model_path);
            std::cerr << "Hand tracking has been enabled." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "FATAL: Could not initialize hand tracker: " << e.what() << std::endl;
            config.track_hands = false; // Disable to prevent crashes
        }
    }
#endif

        // Run selected mode
        if (config.mode == "calibrate") {
            runCalibration(colors);
        } else if (config.mode == "stream") {
            #ifdef ENABLE_HAND_TRACKING
                runStreaming(colors, config, hand_tracker);
            #else
                runStreaming(colors, config);
            #endif
        } else {
            #ifdef ENABLE_HAND_TRACKING
                runTracking(colors, config, hand_tracker);
            #else
                runTracking(colors, config);
            #endif
        }
        
    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}