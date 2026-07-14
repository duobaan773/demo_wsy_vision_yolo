#ifndef RM_STAGE2_DETECTOR_H
#define RM_STAGE2_DETECTOR_H

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class Detector
{
public:
    struct PreprocessInfo
    {
        float scale = 1.0F;
        int pad_left = 0;
        int pad_top = 0;
        cv::Size original_size;
    };

    Detector() = default;

    bool loadModel(const std::string& model_path);

    // 完成 LetterBox、blobFromImage、forward
    bool infer(
        const cv::Mat& image,
        std::vector<cv::Mat>& outputs
    );

    bool isLoaded() const;

    const PreprocessInfo& getPreprocessInfo() const;

private:
    cv::Mat preprocess(const cv::Mat& image);

    void printOutputShapes(
        const std::vector<cv::Mat>& outputs
    ) const;

private:
    cv::dnn::Net net_;
    bool model_loaded_ = false;

    // Ultralytics 默认导出尺寸通常为 640×640
    int input_width_ = 1024;
    int input_height_ = 1024;

    PreprocessInfo preprocess_info_;

    // 只在第一次推理后打印输出形状
    bool output_shape_printed_ = false;
};

#endif