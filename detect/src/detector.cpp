#include "detector.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>

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

        // 当前使用 OpenCV DNN + CPU，兼容性最好
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

    // 使用较小缩放比例，保持原图宽高比
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

        const auto inference_start =
    std::chrono::steady_clock::now();

net_.forward(outputs, output_names);

const auto inference_end =
    std::chrono::steady_clock::now();

inference_time_ms_ =
    std::chrono::duration<double, std::milli>(
        inference_end - inference_start
    ).count();

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

bool Detector::postprocess(
    const std::vector<cv::Mat>& outputs,
    std::vector<Detection>& detections
) const
{
    detections.clear();

    if (outputs.empty())
    {
        std::cerr
            << "[Detector] No output tensor to process.\n";

        return false;
    }

    const cv::Mat& output = outputs.front();

    if (output.empty())
    {
        std::cerr
            << "[Detector] Output tensor is empty.\n";

        return false;
    }

    if (output.dims != 3 || output.size[0] != 1)
    {
        std::cerr
            << "[Detector] Unsupported output tensor shape.\n";

        return false;
    }

    const int expected_attribute_count =
        4 + static_cast<int>(class_names_.size());

    int attribute_count = 0;
    int candidate_count = 0;
    bool channels_first = false;

    /*
     * 支持两种常见输出：
     *
     * [1, 10, 21504]
     * [1, 21504, 10]
     */
    if (output.size[1] == expected_attribute_count)
    {
        channels_first = true;
        attribute_count = output.size[1];
        candidate_count = output.size[2];
    }
    else if (output.size[2] == expected_attribute_count)
    {
        channels_first = false;
        candidate_count = output.size[1];
        attribute_count = output.size[2];
    }
    else
    {
        std::cerr
            << "[Detector] Unexpected attribute count. "
            << "Expected "
            << expected_attribute_count
            << ", tensor shape is "
            << output.size[0] << " x "
            << output.size[1] << " x "
            << output.size[2] << '\n';

        return false;
    }

    if (attribute_count != expected_attribute_count)
    {
        return false;
    }

    cv::Mat continuous_output;

    if (output.isContinuous())
    {
        continuous_output = output;
    }
    else
    {
        continuous_output = output.clone();
    }

    const float* data =
        reinterpret_cast<const float*>(
            continuous_output.data
        );

    std::vector<cv::Rect> candidate_boxes;
    std::vector<float> candidate_scores;
    std::vector<int> candidate_class_ids;
    std::vector<cv::Point2f> candidate_centers;

    candidate_boxes.reserve(candidate_count);
    candidate_scores.reserve(candidate_count);
    candidate_class_ids.reserve(candidate_count);
    candidate_centers.reserve(candidate_count);

    const auto getValue =
        [&](int candidate_index, int attribute_index)
        {
            if (channels_first)
            {
                return data[
                    attribute_index * candidate_count +
                    candidate_index
                ];
            }

            return data[
                candidate_index * attribute_count +
                attribute_index
            ];
        };

    for (int candidate_index = 0;
         candidate_index < candidate_count;
         ++candidate_index)
    {
        float best_score = 0.0F;
        int best_class_id = -1;

        for (int class_id = 0;
             class_id < static_cast<int>(class_names_.size());
             ++class_id)
        {
            const float class_score =
                getValue(
                    candidate_index,
                    4 + class_id
                );

            if (class_score > best_score)
            {
                best_score = class_score;
                best_class_id = class_id;
            }
        }

        if (best_class_id < 0 ||
            best_score < confidence_threshold_)
        {
            continue;
        }

        // 模型输出坐标位于 LetterBox 后的 1024×1024 图像中
        const float center_x_model =
            getValue(candidate_index, 0);

        const float center_y_model =
            getValue(candidate_index, 1);

        const float width_model =
            getValue(candidate_index, 2);

        const float height_model =
            getValue(candidate_index, 3);

        // 从 LetterBox 坐标还原到原图坐标
        const float center_x_original =
            (center_x_model -
             static_cast<float>(preprocess_info_.pad_left)) /
            preprocess_info_.scale;

        const float center_y_original =
            (center_y_model -
             static_cast<float>(preprocess_info_.pad_top)) /
            preprocess_info_.scale;

        const float width_original =
            width_model /
            preprocess_info_.scale;

        const float height_original =
            height_model /
            preprocess_info_.scale;

        float left =
            center_x_original -
            width_original / 2.0F;

        float top =
            center_y_original -
            height_original / 2.0F;

        float right =
            center_x_original +
            width_original / 2.0F;

        float bottom =
            center_y_original +
            height_original / 2.0F;

        // 防止检测框越过图像边界
        left = std::clamp(
            left,
            0.0F,
            static_cast<float>(
                preprocess_info_.original_size.width - 1
            )
        );

        top = std::clamp(
            top,
            0.0F,
            static_cast<float>(
                preprocess_info_.original_size.height - 1
            )
        );

        right = std::clamp(
            right,
            0.0F,
            static_cast<float>(
                preprocess_info_.original_size.width - 1
            )
        );

        bottom = std::clamp(
            bottom,
            0.0F,
            static_cast<float>(
                preprocess_info_.original_size.height - 1
            )
        );

        const int box_x =
            static_cast<int>(std::round(left));

        const int box_y =
            static_cast<int>(std::round(top));

        const int box_width =
            static_cast<int>(
                std::round(right - left)
            );

        const int box_height =
            static_cast<int>(
                std::round(bottom - top)
            );

        // 过滤异常小框或无效框
        if (box_width <= 1 ||
            box_height <= 1)
        {
            continue;
        }

        candidate_boxes.emplace_back(
            box_x,
            box_y,
            box_width,
            box_height
        );

        candidate_scores.push_back(best_score);
        candidate_class_ids.push_back(best_class_id);

        candidate_centers.emplace_back(
            center_x_original,
            center_y_original
        );
    }

    if (candidate_boxes.empty())
    {
        return true;
    }

    /*
     * 分类别执行 NMS。
     *
     * 如果把所有类别一起做 NMS，
     * 不同类别但位置相同的框也可能互相抑制。
     */
    for (int class_id = 0;
         class_id < static_cast<int>(class_names_.size());
         ++class_id)
    {
        std::vector<cv::Rect> class_boxes;
        std::vector<float> class_scores;
        std::vector<int> original_indices;

        for (std::size_t index = 0;
             index < candidate_boxes.size();
             ++index)
        {
            if (candidate_class_ids[index] != class_id)
            {
                continue;
            }

            class_boxes.push_back(
                candidate_boxes[index]
            );

            class_scores.push_back(
                candidate_scores[index]
            );

            original_indices.push_back(
                static_cast<int>(index)
            );
        }

        if (class_boxes.empty())
        {
            continue;
        }

        std::vector<int> kept_indices;

        cv::dnn::NMSBoxes(
            class_boxes,
            class_scores,
            confidence_threshold_,
            nms_threshold_,
            kept_indices
        );

        for (const int class_index : kept_indices)
        {
            const int original_index =
                original_indices[class_index];

            Detection detection;

            detection.class_id =
                candidate_class_ids[original_index];

            detection.confidence =
                candidate_scores[original_index];

            detection.box =
                candidate_boxes[original_index];

            // 使用最终检测框中心，保证中心点处于裁剪后的框内
            detection.center = cv::Point2f(
                static_cast<float>(
                    detection.box.x
                ) +
                static_cast<float>(
                    detection.box.width
                ) / 2.0F,

                static_cast<float>(
                    detection.box.y
                ) +
                static_cast<float>(
                    detection.box.height
                ) / 2.0F
            );

            detections.push_back(detection);
        }
    }

    // 按置信度从高到低排序，便于显示和目标筛选
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& first,
           const Detection& second)
        {
            return first.confidence >
                   second.confidence;
        }
    );

    return true;
}

bool Detector::detect(
    const cv::Mat& image,
    std::vector<Detection>& detections
)
{
    detections.clear();

    std::vector<cv::Mat> outputs;

    if (!infer(image, outputs))
    {
        return false;
    }

    const auto postprocess_start =
        std::chrono::steady_clock::now();

    const bool postprocess_success =
        postprocess(outputs, detections);

    const auto postprocess_end =
        std::chrono::steady_clock::now();

    postprocess_time_ms_ =
        std::chrono::duration<double, std::milli>(
            postprocess_end - postprocess_start
        ).count();

    return postprocess_success;
}
{
    detections.clear();

    std::vector<cv::Mat> outputs;

    if (!infer(image, outputs))
    {
        return false;
    }

    return postprocess(
        outputs,
        detections
    );
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
        const cv::Mat& output =
            outputs[index];

        std::cout
            << "[Detector] Output[" << index
            << "] shape: ";

        for (int dimension = 0;
             dimension < output.dims;
             ++dimension)
        {
            std::cout
                << output.size[dimension];

            if (dimension + 1 < output.dims)
            {
                std::cout << " x ";
            }
        }

        std::cout
            << ", type="
            << output.type()
            << '\n';
    }
}

bool Detector::isLoaded() const
{
    return model_loaded_ &&
           !net_.empty();
}

const std::vector<std::string>&
Detector::getClassNames() const
{
    return class_names_;
}

const Detector::PreprocessInfo&
Detector::getPreprocessInfo() const
{
    return preprocess_info_;
}
double Detector::getInferenceTimeMs() const
{
    return inference_time_ms_;
}

double Detector::getPostprocessTimeMs() const
{
    return postprocess_time_ms_;
}