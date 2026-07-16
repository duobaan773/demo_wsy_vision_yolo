#ifndef RM_STAGE2_INPUT_HANDLER_H
#define RM_STAGE2_INPUT_HANDLER_H

#include <opencv2/opencv.hpp>

#include <string>

class InputHandler
{
public:
    double getSourceFPS() const;
    enum class InputType
    {
        None,
        Image,
        Video,
        Camera
    };

    InputHandler() = default;
    ~InputHandler();

    // 禁止复制，避免 VideoCapture 等资源被重复管理
    InputHandler(const InputHandler&) = delete;
    InputHandler& operator=(const InputHandler&) = delete;

    bool openImage(const std::string& image_path);
    bool openVideo(const std::string& video_path);
    bool openCamera(int camera_id = 0);

    // 统一获取一帧。
    // 图片只返回一次；视频和摄像头逐帧返回。
    bool readFrame(cv::Mat& frame);

    void release();

    InputType getInputType() const;
    bool isOpened() const;

private:
    InputType input_type_ = InputType::None;

    cv::Mat image_;
    bool image_has_been_read_ = false;

    cv::VideoCapture capture_;
};

#endif