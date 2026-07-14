#ifndef RM_STAGE2_DETECTOR_H
#define RM_STAGE2_DETECTOR_H

#include <opencv2/dnn.hpp>

#include <string>

class Detector
{
public:
    Detector() = default;

    // 加载 ONNX 模型并配置 OpenCV DNN 后端
    bool loadModel(const std::string& model_path);

    bool isLoaded() const;

private:
    cv::dnn::Net net_;
    bool model_loaded_ = false;
};

#endif