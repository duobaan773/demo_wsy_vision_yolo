#ifndef RM_STAGE2_VISUALIZER_H
#define RM_STAGE2_VISUALIZER_H

#include "detector.h"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

class Visualizer
{
public:
    void drawPerformance(
    cv::Mat& image,
    double fps,
    double average_fps,
    double inference_time_ms,
    double postprocess_time_ms
) const;
    Visualizer() = default;

    void drawDetections(
        cv::Mat& image,
        const std::vector<Detection>& detections,
        const std::vector<std::string>& class_names
    ) const;

private:
    static cv::Scalar getClassColor(int class_id);
};

#endif