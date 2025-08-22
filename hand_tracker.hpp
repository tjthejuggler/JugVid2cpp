#pragma once

#ifdef ENABLE_HAND_TRACKING // Only compile this file if the feature is enabled

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

// Simple landmark structure to match MediaPipe's interface
struct NormalizedLandmark {
    float x, y, z;
    NormalizedLandmark(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}
};

// Struct to hold simplified results for a detected hand
struct HandResult {
    std::vector<NormalizedLandmark> normalized_landmarks;
};

class HandTracker {
public:
    // Constructor - OpenCV-based hand tracking doesn't need a model file
    HandTracker(const std::string& model_path = "") {
        // Initialize OpenCV's HOG descriptor for hand detection
        // Note: OpenCV doesn't have built-in hand landmark detection like MediaPipe
        // This is a simplified implementation that detects hand regions
        hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
        
        // Initialize background subtractor for motion detection
        bg_subtractor_ = cv::createBackgroundSubtractorMOG2(500, 16, true);
        
        std::cerr << "OpenCV-based hand tracker initialized (simplified implementation)" << std::endl;
    }

    // Processes a single cv::Mat frame and returns detected hands
    std::vector<HandResult> DetectHands(const cv::Mat& frame, int64_t timestamp_ms) {
        std::vector<HandResult> detected_hands;
        
        if (frame.empty()) {
            return detected_hands;
        }
        
        try {
            // Convert to grayscale for processing
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            
            // Apply background subtraction to detect moving objects (hands)
            cv::Mat fg_mask;
            bg_subtractor_->apply(frame, fg_mask);
            
            // Find contours in the foreground mask
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            // Filter contours by size to find potential hand regions
            for (const auto& contour : contours) {
                double area = cv::contourArea(contour);
                if (area > 1000 && area < 50000) { // Reasonable hand size range
                    // Get bounding rectangle
                    cv::Rect hand_rect = cv::boundingRect(contour);
                    
                    // Create a simplified hand result with key points
                    HandResult hand;
                    
                    // Generate simplified landmarks (5 key points: palm center + 4 fingertips approximation)
                    float center_x = (hand_rect.x + hand_rect.width / 2.0f) / frame.cols;
                    float center_y = (hand_rect.y + hand_rect.height / 2.0f) / frame.rows;
                    
                    // Palm center
                    hand.normalized_landmarks.push_back(NormalizedLandmark(center_x, center_y, 0.0f));
                    
                    // Approximate fingertip positions (simplified)
                    hand.normalized_landmarks.push_back(NormalizedLandmark(center_x - 0.02f, center_y - 0.05f, 0.0f)); // Index
                    hand.normalized_landmarks.push_back(NormalizedLandmark(center_x, center_y - 0.06f, 0.0f)); // Middle
                    hand.normalized_landmarks.push_back(NormalizedLandmark(center_x + 0.02f, center_y - 0.05f, 0.0f)); // Ring
                    hand.normalized_landmarks.push_back(NormalizedLandmark(center_x + 0.04f, center_y - 0.02f, 0.0f)); // Pinky
                    
                    detected_hands.push_back(hand);
                    
                    // Limit to 2 hands maximum
                    if (detected_hands.size() >= 2) break;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Hand detection error: " << e.what() << std::endl;
        }
        
        return detected_hands;
    }

private:
    cv::HOGDescriptor hog_;
    cv::Ptr<cv::BackgroundSubtractor> bg_subtractor_;
};

#endif // ENABLE_HAND_TRACKING