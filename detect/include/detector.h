#ifndef RM_STAGE2_DETECTOR_H
#define RM_STAGE2_DETECTOR_H

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

// 单个检测目标的最终结果
struct Detection
{
    int class_id = -1;
    float confidence = 0.0F;

    // 映射回原始图像后的检测框
    cv::Rect box;

    // 装甲板中心点
    cv::Point2f center;
};

class Detector
{
public:
double getInferenceTimeMs() const;
double getPostprocessTimeMs() const;
    struct PreprocessInfo
    {
        // 原图缩放到模型输入尺寸时的比例
        float scale = 1.0F;

        // LetterBox 左侧和顶部填充
        int pad_left = 0;
        int pad_top = 0;

        cv::Size original_size;
    };

    Detector() = default;

    // 加载 ONNX 模型
    bool loadModel(const std::string& model_path);

    // 完成前处理、forward、后处理和 NMS
    bool detect(
        const cv::Mat& image,
        std::vector<Detection>& detections
    );

    bool isLoaded() const;

    const std::vector<std::string>& getClassNames() const;

    const PreprocessInfo& getPreprocessInfo() const;

private:
    // LetterBox + blobFromImage
    cv::Mat preprocess(const cv::Mat& image);

    // 网络前向推理
    bool infer(
        const cv::Mat& image,
        std::vector<cv::Mat>& outputs
    );

    // 解析输出、坐标还原、NMS、中心点计算
    bool postprocess(
        const std::vector<cv::Mat>& outputs,
        std::vector<Detection>& detections
    ) const;

    void printOutputShapes(
        const std::vector<cv::Mat>& outputs
    ) const;

private:
    double inference_time_ms_ = 0.0;
    double postprocess_time_ms_ = 0.0;
    cv::dnn::Net net_;
    bool model_loaded_ = false;

    // 你的 ONNX 模型固定输入尺寸
    int input_width_ = 1024;
    int input_height_ = 1024;

    // 置信度和 NMS 阈值
    float confidence_threshold_ = 0.50F;
    float nms_threshold_ = 0.45F;

    PreprocessInfo preprocess_info_;

    bool output_shape_printed_ = false;

    // 必须严格对应 data.yaml 中的类别顺序
    std::vector<std::string> class_names_ = {
        "bluesb",
        "redsb",
        "blue3",
        "red3",
        "blue1",
        "red1"
    };
};

#endif