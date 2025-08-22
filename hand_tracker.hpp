#pragma once

#ifdef ENABLE_HAND_TRACKING // Only compile this file if the feature is enabled

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include <string>
#include <vector>

// Struct to hold simplified results for a detected hand
struct HandResult {
    // We need both normalized and world landmarks.
    // Normalized landmarks for getting 3D position from RealSense depth data.
    // World landmarks for relative distance checks if needed.
    std::vector<mediapipe::NormalizedLandmark> normalized_landmarks;
};

class HandTracker {
public:
    // Constructor initializes the MediaPipe Hand Landmarker task
    HandTracker(const std::string& model_path) {
        auto options = std::make_unique<mediapipe::tasks::vision::hand_landmarker::HandLandmarkerOptions>();
        options->base_options.model_asset_path = model_path;
        options->num_hands = 2;
        options->running_mode = mediapipe::tasks::vision::RunningMode::VIDEO;

        auto landmarker_or_status = mediapipe::tasks::vision::hand_landmarker::HandLandmarker::Create(std::move(options));
        if (!landmarker_or_status.ok()) {
            throw std::runtime_error("Failed to create MediaPipe Hand Landmarker: " + landmarker_or_status.status().ToString());
        }
        m_landmarker = std::move(landmarker_or_status.value());
    }

    // Processes a single cv::Mat frame and returns detected hands
    std::vector<HandResult> DetectHands(const cv::Mat& frame, int64_t timestamp_ms) {
        // Convert cv::Mat to MediaPipe's Image format
        auto mp_image = std::make_shared<mediapipe::Image>(
            mediapipe::ImageFormat::SRGB, frame.cols, frame.rows,
            frame.step, frame.data, nullptr);

        // Process the image
        auto results_or_status = m_landmarker->DetectForVideo(*mp_image, timestamp_ms);
        if (!results_or_status.ok()) {
            std::cerr << "Hand detection failed: " << results_or_status.status().ToString() << std::endl;
            return {};
        }

        // Convert the results into our simplified struct
        std::vector<HandResult> detected_hands;
        for (const auto& hand_landmarks : results_or_status.value().hand_landmarks) {
            HandResult hand;
            hand.normalized_landmarks = hand_landmarks;
            detected_hands.push_back(hand);
        }
        return detected_hands;
    }

private:
    std::unique_ptr<mediapipe::tasks::vision::hand_landmarker::HandLandmarker> m_landmarker;
};

#endif // ENABLE_HAND_TRACKING