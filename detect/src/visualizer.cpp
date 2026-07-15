#include "visualizer.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

cv::Scalar Visualizer::getClassColor(int class_id)
{
    // 保留类别颜色接口，当前最终显示统一使用绿色框。
    // 后续需要按红蓝阵营区分时可以重新启用。
    switch (class_id)
    {
    case 0: // bluesb
    case 2: // blue3
    case 4: // blue1
        return cv::Scalar(255, 0, 0);

    case 1: // redsb
    case 3: // red3
    case 5: // red1
        return cv::Scalar(0, 0, 255);

    default:
        return cv::Scalar(0, 255, 0);
    }
}

void Visualizer::drawDetections(
    cv::Mat& image,
    const std::vector<Detection>& detections,
    const std::vector<std::string>& class_names
) const
{
    for (const Detection& detection : detections)
    {
        // 为了验收时显示更清楚，检测框统一使用绿色。
        const cv::Scalar box_color(0, 255, 0);

        // 1. 绘制检测框
        cv::rectangle(
            image,
            detection.box,
            box_color,
            4
        );

        // 2. 绘制装甲板中心点
        const cv::Point center(
            cvRound(detection.center.x),
            cvRound(detection.center.y)
        );

        // 红色实心圆点
        cv::circle(
            image,
            center,
            8,
            cv::Scalar(0, 0, 255),
            cv::FILLED
        );

        // 红色十字准星
        cv::line(
            image,
            cv::Point(center.x - 15, center.y),
            cv::Point(center.x + 15, center.y),
            cv::Scalar(0, 0, 255),
            2
        );

        cv::line(
            image,
            cv::Point(center.x, center.y - 15),
            cv::Point(center.x, center.y + 15),
            cv::Scalar(0, 0, 255),
            2
        );

        // 3. 生成“类别 + 置信度”标签
        std::ostringstream label_stream;
        label_stream << std::fixed
                     << std::setprecision(2);

        if (detection.class_id >= 0 &&
            detection.class_id <
                static_cast<int>(class_names.size()))
        {
            label_stream << class_names[detection.class_id];
        }
        else
        {
            label_stream << "unknown";
        }

        label_stream << " "
                     << detection.confidence;

        const std::string label =
            label_stream.str();

        // 4. 计算文字尺寸
        int baseline = 0;

        const cv::Size text_size =
            cv::getTextSize(
                label,
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
                &baseline
            );

        const int label_left =
            detection.box.x;

        const int label_top =
            std::max(
                detection.box.y,
                text_size.height + 10
            );

        // 5. 绘制标签背景
        cv::rectangle(
            image,
            cv::Point(
                label_left,
                label_top - text_size.height - 8
            ),
            cv::Point(
                label_left + text_size.width + 8,
                label_top
            ),
            box_color,
            cv::FILLED
        );

        // 6. 绘制标签文字
        cv::putText(
            image,
            label,
            cv::Point(
                label_left + 4,
                label_top - 4
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.6,
            cv::Scalar(0, 0, 0),
            2
        );

        // 7. 在中心点旁边显示坐标
        std::ostringstream center_stream;
        center_stream
            << "("
            << center.x
            << ", "
            << center.y
            << ")";

        cv::putText(
            image,
            center_stream.str(),
            cv::Point(
                center.x + 12,
                center.y - 12
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 0, 255),
            2
        );
    }

    // 8. 左上角显示检测数量
    cv::putText(
        image,
        "Detections: " +
            std::to_string(detections.size()),
        cv::Point(20, 40),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 255, 0),
        2
    );
}
void Visualizer::drawPerformance(
    cv::Mat& image,
    double fps,
    double average_fps,
    double inference_time_ms,
    double postprocess_time_ms
) const
{
    std::ostringstream performance_stream;

    performance_stream
        << std::fixed
        << std::setprecision(1)
        << "FPS: " << fps
        << "  AVG: " << average_fps
        << "  Infer: " << inference_time_ms << " ms"
        << "  Post: " << postprocess_time_ms << " ms";

    cv::putText(
        image,
        performance_stream.str(),
        cv::Point(20, 75),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65,
        cv::Scalar(0, 255, 0),
        2
    );
}
