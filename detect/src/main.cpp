#include "input_handler.h"

#include <opencv2/opencv.hpp>

#include <exception>
#include <iostream>
#include <string>

namespace
{

void printUsage(const char* program_name)
{
    std::cout
        << "Usage:\n"
        << "  " << program_name << " image  <image_path>\n"
        << "  " << program_name << " video  <video_path>\n"
        << "  " << program_name << " camera [camera_id]\n\n"
        << "Examples:\n"
        << "  " << program_name << " image assets/test.jpg\n"
        << "  " << program_name << " video assets/test.mp4\n"
        << "  " << program_name << " camera 0\n";
}

bool openInputFromArguments(
    int argc,
    char* argv[],
    InputHandler& input_handler)
{
    if (argc < 2)
    {
        std::cerr << "[Main] Missing input mode.\n";
        return false;
    }

    const std::string mode = argv[1];

    if (mode == "image")
    {
        if (argc < 3)
        {
            std::cerr << "[Main] Missing image path.\n";
            return false;
        }

        return input_handler.openImage(argv[2]);
    }

    if (mode == "video")
    {
        if (argc < 3)
        {
            std::cerr << "[Main] Missing video path.\n";
            return false;
        }

        return input_handler.openVideo(argv[2]);
    }

    if (mode == "camera")
    {
        int camera_id = 0;

        if (argc >= 3)
        {
            try
            {
                camera_id = std::stoi(argv[2]);
            }
            catch (const std::exception&)
            {
                std::cerr << "[Main] Invalid camera ID: "
                          << argv[2] << '\n';
                return false;
            }
        }

        return input_handler.openCamera(camera_id);
    }

    std::cerr << "[Main] Unsupported input mode: "
              << mode << '\n';

    return false;
}

} // namespace

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << " RoboMaster 2027 Stage2 Detect System\n";
    std::cout << "========================================\n";

    InputHandler input_handler;

    if (!openInputFromArguments(argc, argv, input_handler))
    {
        printUsage(argv[0]);
        return 1;
    }

    cv::Mat frame;

    while (input_handler.readFrame(frame))
    {
        /*
         * 后续最终处理链路插入在这里：
         *
         * 1. detector.detect(frame)
         * 2. YOLO 后处理与 NMS
         * 3. 计算所有装甲板中心点
         * 4. visualizer.draw(...)
         * 5. 显示 FPS、推理耗时和后处理耗时
         */

        cv::putText(
            frame,
            "Input OK",
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        cv::imshow("RM Stage2 Detection", frame);

        // 图片等待任意按键；视频和摄像头每帧等待 1 ms
        const int delay =
            input_handler.getInputType() == InputHandler::InputType::Image
                ? 0
                : 1;

        const int key = cv::waitKey(delay);

        // Esc 或 q 退出
        if (key == 27 || key == 'q' || key == 'Q')
        {
            break;
        }
    }

    input_handler.release();
    cv::destroyAllWindows();

    std::cout << "[Main] Program exited normally.\n";
    return 0;
}