#include <iostream>

#include "detector.h"
#include "input_handler.h"
#include "timer.h"
#include "visualizer.h"

int main(int argc, char* argv[])
{
    std::cout << "========================================\n";
    std::cout << " RoboMaster 2027 Stage2 Detect System\n";
    std::cout << "========================================\n";

    /*
     * 最终运行形式：
     *
     * rm_detect image  assets/test.jpg
     * rm_detect video  assets/test.mp4
     * rm_detect camera 0
     *
     * 当前提交只建立工程骨架。
     * 输入解析和具体处理将在后续功能提交中实现。
     */

    InputHandler input_handler;
    Detector detector;
    Visualizer visualizer;
    Timer timer;

    if (!input_handler.initialize() ||
        !detector.initialize() ||
        !visualizer.initialize())
    {
        std::cerr << "Failed to initialize project modules.\n";
        return 1;
    }

    timer.reset();

    std::cout << "Project architecture initialized successfully.\n";
    return 0;
}