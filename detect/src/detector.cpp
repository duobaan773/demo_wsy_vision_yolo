#include "detector.h"

#include <algorithm>
#include <cmath>
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
            std::cerr
                << "[Detector] Loaded network is empty: "
                << model_path << '\n';

            return false;
        }

        net_.setPreferableBackend(
            cv::dnn::DNN_BACKEND_OPENCV
        );

        net_.setPreferableTarget(
            cv::dnn::DNN_TARGET_CPU
        );

        model_loaded_ = true;

        std::cout
            << "[Detector] ONNX model loaded successfully: "
            << model_path << '\n';

        std::cout
            << "[Detector] Number of network layers: "
            << net_.getLayerNames().size() << '\n';

        std::cout
            << "[Detector] Input size: "
            << input_width_ << " x "
            << input_height_ << '\n';

        std::cout
            << "[Detector] Backend: OpenCV DNN\n";

        std::cout
            << "[Detector] Target: CPU\n";

        return true;
    }
    catch (const cv::Exception& exception)
    {
        std::cerr
            << "[Detector] Failed to load ONNX model.\n";

        std::cerr
            << "[Detector] Model path: "
            << model_path << '\n';

        std::cerr
            << "[Detector] OpenCV error: "
            << exception.what() << '\n';

        return false;
    }
}

cv::Mat Detector::preprocess(const cv::Mat& image)
{
    if (image.empty())
    {
        return {};
    }

    preprocess_info_.original_size = image.size();

    const float scale_x =
        static_cast<float>(input_width_) /
        static_cast<float>(image.cols);

    const float scale_y =
        static_cast<float>(input_height_) /
        static_cast<float>(image.rows);

    // 保持原始宽高比
    preprocess_info_.scale =
        std::min(scale_x, scale_y);

    const int resized_width =
        static_cast<int>(
            std::round(
                static_cast<float>(image.cols) *
                preprocess_info_.scale
            )
        );

    const int resized_height =
        static_cast<int>(
            std::round(
                static_cast<float>(image.rows) *
                preprocess_info_.scale
            )
        );

    cv::Mat resized_image;

    cv::resize(
        image,
        resized_image,
        cv::Size(resized_width, resized_height),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    const int total_pad_x =
        input_width_ - resized_width;

    const int total_pad_y =
        input_height_ - resized_height;

    preprocess_info_.pad_left =
        total_pad_x / 2;

    preprocess_info_.pad_top =
        total_pad_y / 2;

    const int pad_right =
        total_pad_x - preprocess_info_.pad_left;

    const int pad_bottom =
        total_pad_y - preprocess_info_.pad_top;

    cv::Mat letterboxed_image;

    cv::copyMakeBorder(
        resized_image,
        letterboxed_image,
        preprocess_info_.pad_top,
        pad_bottom,
        preprocess_info_.pad_left,
        pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    cv::Mat blob;

    cv::dnn::blobFromImage(
        letterboxed_image,
        blob,
        1.0 / 255.0,
        cv::Size(input_width_, input_height_),
        cv::Scalar(),
        true,   // BGR 转 RGB
        false,  // 不裁剪
        CV_32F
    );

    return blob;
}

bool Detector::infer(
    const cv::Mat& image,
    std::vector<cv::Mat>& outputs
)
{
    outputs.clear();

    if (!isLoaded())
    {
        std::cerr
            << "[Detector] Model has not been loaded.\n";

        return false;
    }

    if (image.empty())
    {
        std::cerr
            << "[Detector] Input image is empty.\n";

        return false;
    }

    try
    {
        const cv::Mat blob = preprocess(image);

        if (blob.empty())
        {
            std::cerr
                << "[Detector] Failed to create input blob.\n";

            return false;
        }

        net_.setInput(blob);

        const std::vector<std::string> output_names =
            net_.getUnconnectedOutLayersNames();

        net_.forward(outputs, output_names);

        if (outputs.empty())
        {
            std::cerr
                << "[Detector] Network output is empty.\n";

            return false;
        }

        if (!output_shape_printed_)
        {
            std::cout
                << "[Detector] LetterBox scale: "
                << preprocess_info_.scale << '\n';

            std::cout
                << "[Detector] LetterBox padding: left="
                << preprocess_info_.pad_left
                << ", top="
                << preprocess_info_.pad_top
                << '\n';

            printOutputShapes(outputs);
            output_shape_printed_ = true;
        }

        return true;
    }
    catch (const cv::Exception& exception)
    {
        std::cerr
            << "[Detector] Inference failed: "
            << exception.what() << '\n';

        return false;
    }
}

void Detector::printOutputShapes(
    const std::vector<cv::Mat>& outputs
) const
{
    std::cout
        << "[Detector] Output tensor count: "
        << outputs.size() << '\n';

    for (std::size_t index = 0;
         index < outputs.size();
         ++index)
    {
        const cv::Mat& output = outputs[index];

        std::cout
            << "[Detector] Output[" << index
            << "] shape: ";

        for (int dimension = 0;
             dimension < output.dims;
             ++dimension)
        {
            std::cout << output.size[dimension];

            if (dimension + 1 < output.dims)
            {
                std::cout << " x ";
            }
        }

        std::cout
            << ", type=" << output.type()
            << '\n';
    }
}

bool Detector::isLoaded() const
{
    return model_loaded_ && !net_.empty();
}

const Detector::PreprocessInfo&
Detector::getPreprocessInfo() const
{
    return preprocess_info_;
}