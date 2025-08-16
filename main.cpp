#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include "json.hpp" // For JSON settings file
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

// Use nlohmann::json for convenience
using json = nlohmann::json;

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

// Unified function to detect balls for a given color range with merging
void detectBalls(const cv::Mat& hsv_frame, const ColorRange& color, std::vector<cv::Point2f>& centers) {
    cv::Mat mask;
    cv::inRange(hsv_frame, color.min_hsv, color.max_hsv, mask);
    
    // Handle second HSV range if valid
    if (color.min_hsv2[0] >= 0 && color.max_hsv2[0] >= 0) {
        cv::Mat mask2;
        cv::inRange(hsv_frame, color.min_hsv2, color.max_hsv2, mask2);
        cv::bitwise_or(mask, mask2, mask);
    }
    
    // Morphological operations to clean up the mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Process contours to get initial centers
    std::vector<cv::Point2f> initial_centers;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) > MIN_CONTOUR_AREA) {
            cv::Moments m = cv::moments(contour);
            if (m.m00 > 0) {
                initial_centers.push_back(cv::Point2f(m.m10 / m.m00, m.m01 / m.m00));
            }
        }
    }
    
    // Merge nearby detections to handle occlusion
    centers = mergeNearbyDetections(initial_centers);
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

// Main tracking logic
void runTracking(std::vector<ColorRange>& colors) {
    std::cerr << "Starting tracking mode..." << std::endl;
    
    // Initialize RealSense pipeline
    rs2::pipeline pipe;
    rs2::config cfg;
    
    // Configure streams with fallback
    int width = 848, height = 480;
    try {
        cfg.enable_stream(RS2_STREAM_COLOR, 848, 480, RS2_FORMAT_BGR8, 90);
        cfg.enable_stream(RS2_STREAM_DEPTH, 848, 480, RS2_FORMAT_Z16, 90);
        auto profile = pipe.start(cfg);
        std::cerr << "Started at 848x480 @ 90 FPS" << std::endl;
    } catch (const rs2::error& e) {
        std::cerr << "848x480 @ 90 FPS not available, trying 1280x720 @ 30 FPS: " << e.what() << std::endl;
        cfg.disable_all_streams();
        cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
        auto profile = pipe.start(cfg);
        width = 1280;
        height = 720;
        std::cerr << "Started at 1280x720 @ 30 FPS" << std::endl;
    }

    // Get camera intrinsics and create alignment object
    auto profile = pipe.get_active_profile();
    auto intrinsics = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    rs2::align align_to_color(RS2_STREAM_COLOR);

    std::cerr << "Ball tracker initialized. Press Ctrl+C to stop." << std::endl;
    
    while (true) {
        rs2::frameset frames;
        if (!pipe.try_wait_for_frames(&frames, 100)) {
            continue;
        }
        
        auto aligned_frames = align_to_color.process(frames);
        auto color_frame = aligned_frames.get_color_frame();
        auto depth_frame = aligned_frames.get_depth_frame();
        
        if (!color_frame || !depth_frame) continue;

        // Convert frames to OpenCV matrices
        cv::Mat color_image(cv::Size(width, height), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat hsv_image;
        cv::cvtColor(color_image, hsv_image, cv::COLOR_BGR2HSV);

        std::vector<BallDetection> all_detections;
        
        // Detect balls for each color
        for (const auto& color : colors) {
            std::vector<cv::Point2f> centers;
            detectBalls(hsv_image, color, centers);

            for (const auto& center : centers) {
                int x = static_cast<int>(center.x);
                int y = static_cast<int>(center.y);
                
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    float depth = depth_frame.get_distance(x, y);
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

        // Output results in the specified format
        if (!all_detections.empty()) {
            std::string output_line;
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

int main(int argc, char* argv[]) {
    try {
        // Default color values optimized for best separation: pink, orange, green, yellow
        // Note: Pink and orange are close in HSV space - for best results use yellow, pink, green
        std::vector<ColorRange> colors = {
            ColorRange("pink", cv::Scalar(150, 150, 90), cv::Scalar(170, 255, 255)),    // Higher hue, more saturated pink
            ColorRange("orange", cv::Scalar(5, 150, 120), cv::Scalar(15, 255, 255)),    // Lower hue, more saturated orange
            ColorRange("green", cv::Scalar(45, 120, 70), cv::Scalar(75, 255, 255)),     // Reliable green range
            ColorRange("yellow", cv::Scalar(25, 120, 100), cv::Scalar(35, 255, 255))    // Reliable yellow range
        };
        
        // Load settings from file, if it exists
        loadSettings(colors);

        // Check for command-line argument to determine mode
        if (argc > 1 && std::string(argv[1]) == "calibrate") {
            runCalibration(colors);
        } else {
            runTracking(colors);
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