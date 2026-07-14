#include "detector.h"
#include "input_handler.h"

#include <opencv2/opencv.hpp>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace
{

constexpr const char* DEFAULT_MODEL_PATH = "../models/best.onnx";

void printUsage(const char* program_name)
{
    std::cout
        << "Usage:\n"
        << "  " << program_name
        << " image  <image_path> [model_path]\n"
        << "  " << program_name
        << " video  <video_path> [model_path]\n"
        << "  " << program_name
        << " camera [camera_id]  [model_path]\n\n"

        << "Examples:\n"
        << "  " << program_name
        << " image assets/test.jpg\n"
        << "  " << program_name
        << " video assets/test.mp4\n"
        << "  " << program_name
        << " camera 0\n"
        << "  " << program_name
        << " image assets/test.jpg ../models/best.onnx\n";
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

std::string getModelPathFromArguments(
    int argc,
    char* argv[])
{
    if (argc < 2)
    {
        return DEFAULT_MODEL_PATH;
    }

    const std::string mode = argv[1];

    // image/video:
    // rm_detect image path [model_path]
    if ((mode == "image" || mode == "video") && argc >= 4)
    {
        return argv[3];
    }

    // camera:
    // rm_detect camera camera_id [model_path]
    if (mode == "camera" && argc >= 4)
    {
        return argv[3];
    }

    return DEFAULT_MODEL_PATH;
}

} // namespace

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << " RoboMaster 2027 Stage2 Detect System\n";
    std::cout << "========================================\n";

    if (argc < 2)
    {
        printUsage(argv[0]);
        return 1;
    }

    const std::string model_path =
        getModelPathFromArguments(argc, argv);

    Detector detector;

    if (!detector.loadModel(model_path))
    {
        std::cerr << "[Main] Failed to initialize detector.\n";
        return 1;
    }

    InputHandler input_handler;

    if (!openInputFromArguments(
            argc,
            argv,
            input_handler))
    {
        printUsage(argv[0]);
        return 1;
    }

    cv::Mat frame;

    while (input_handler.readFrame(frame))
    {
       std::vector<cv::Mat> outputs;

if (!detector.infer(frame, outputs))
{
    std::cerr << "[Main] Inference failed.\n";
    break;
}

cv::putText(
    frame,
    "YOLO Inference OK",
    cv::Point(20, 40),
    cv::FONT_HERSHEY_SIMPLEX,
    1.0,
    cv::Scalar(0, 255, 0),
    2
);

      
        cv::imshow(
            "RM Stage2 Detection",
            frame
        );

        const int delay =
            input_handler.getInputType()
                    == InputHandler::InputType::Image
                ? 0
                : 1;

        const int key = cv::waitKey(delay);

        if (key == 27 ||
            key == 'q' ||
            key == 'Q')
        {
            break;
        }
    }

    input_handler.release();
    cv::destroyAllWindows();

    std::cout << "[Main] Program exited normally.\n";
    return 0;
}