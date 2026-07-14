#include "detector.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>

bool Detector::loadModel(const std::string& model_path)
{
    model_loaded_ = false;
    net_ = cv::dnn::Net();

    try
    {
        net_ = cv::dnn::readNetFromONNX(model_path);

        if (net_.empty())
        {
            std::cerr << "[Detector] Loaded network is empty: "
                      << model_path << '\n';
            return false;
        }

        // 当前先使用 OpenCV 自带后端和 CPU。
        // 兼容性最好，符合阶段二基础部署要求。
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        model_loaded_ = true;

        std::cout << "[Detector] ONNX model loaded successfully: "
                  << model_path << '\n';

        std::cout << "[Detector] Number of network layers: "
                  << net_.getLayerNames().size() << '\n';

        std::cout << "[Detector] Backend: OpenCV DNN\n";
        std::cout << "[Detector] Target: CPU\n";

        return true;
    }
    catch (const cv::Exception& exception)
    {
        std::cerr << "[Detector] OpenCV failed to load ONNX model.\n";
        std::cerr << "[Detector] Model path: " << model_path << '\n';
        std::cerr << "[Detector] Error: " << exception.what() << '\n';
        return false;
    }
}

bool Detector::isLoaded() const
{
    return model_loaded_ && !net_.empty();
}