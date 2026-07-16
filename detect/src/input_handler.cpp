#include "input_handler.h"

#include <iostream>

InputHandler::~InputHandler()
{
    release();
}

bool InputHandler::openImage(const std::string& image_path)
{
    release();

    image_ = cv::imread(image_path);

    if (image_.empty())
    {
        std::cerr << "[InputHandler] Failed to open image: "
                  << image_path << '\n';
        return false;
    }

    input_type_ = InputType::Image;
    image_has_been_read_ = false;

    std::cout << "[InputHandler] Image opened successfully: "
              << image_path << '\n';

    return true;
}

bool InputHandler::openVideo(const std::string& video_path)
{
    release();

    // 显式指定 FFmpeg 后端，避免 OpenCV 自动选择到不合适的后端
    if (!capture_.open(video_path, cv::CAP_FFMPEG))
    {
        std::cerr << "[InputHandler] Failed to open video with FFmpeg: "
                  << video_path << '\n';
        return false;
    }

    input_type_ = InputType::Video;

    std::cout << "[InputHandler] Video opened successfully: "
              << video_path << '\n';

    std::cout << "[InputHandler] Backend: "
              << capture_.getBackendName() << '\n';

    std::cout << "[InputHandler] Resolution: "
              << static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH))
              << " x "
              << static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT))
              << '\n';

    std::cout << "[InputHandler] Source FPS: "
              << capture_.get(cv::CAP_PROP_FPS) << '\n';

    std::cout << "[InputHandler] Frame count: "
              << static_cast<long long>(
                     capture_.get(cv::CAP_PROP_FRAME_COUNT))
              << '\n';

    return true;
}
bool InputHandler::openCamera(int camera_id)
{
    release();

    if (!capture_.open(camera_id))
    {
        std::cerr << "[InputHandler] Failed to open camera ID: "
                  << camera_id << '\n';
        return false;
    }

    input_type_ = InputType::Camera;

    std::cout << "[InputHandler] Camera opened successfully. ID: "
              << camera_id << '\n';

    return true;
}

bool InputHandler::readFrame(cv::Mat& frame)
{
    frame.release();

    switch (input_type_)
    {
    case InputType::Image:
        // 单张图片只向主流程提供一次
        if (image_has_been_read_ || image_.empty())
        {
            return false;
        }

        frame = image_.clone();
        image_has_been_read_ = true;
        return true;

    case InputType::Video:
    case InputType::Camera:
        if (!capture_.isOpened())
        {
            return false;
        }

        capture_ >> frame;
        return !frame.empty();

    case InputType::None:
    default:
        return false;
    }
}

void InputHandler::release()
{
    if (capture_.isOpened())
    {
        capture_.release();
    }

    image_.release();
    image_has_been_read_ = false;
    input_type_ = InputType::None;
}

InputHandler::InputType InputHandler::getInputType() const
{
    return input_type_;
}

bool InputHandler::isOpened() const
{
    switch (input_type_)
    {
    case InputType::Image:
        return !image_.empty();

    case InputType::Video:
    case InputType::Camera:
        return capture_.isOpened();

    case InputType::None:
    default:
        return false;
    }
}
double InputHandler::getSourceFPS() const
{
    if (input_type_ == InputType::Video &&
        capture_.isOpened())
    {
        const double fps =
            capture_.get(cv::CAP_PROP_FPS);

        return fps > 0.0 ? fps : 30.0;
    }

    return 30.0;
}