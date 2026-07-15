#ifndef RM_STAGE2_TIMER_H
#define RM_STAGE2_TIMER_H

#include <chrono>

class Timer
{
public:
    using Clock = std::chrono::steady_clock;

    Timer() = default;

    // 在每帧处理开始时调用
    void start();

    // 在每帧处理结束时调用
    void stop();

    // 本帧总耗时，单位 ms
    double getFrameTimeMs() const;

    // 根据本帧总耗时计算的瞬时 FPS
    double getFPS() const;

    // 从程序开始到现在的平均 FPS
    double getAverageFPS() const;

    void reset();

private:
    Clock::time_point start_time_{};

    double frame_time_ms_ = 0.0;
    double total_time_ms_ = 0.0;

    long long frame_count_ = 0;
    bool running_ = false;
};

#endif