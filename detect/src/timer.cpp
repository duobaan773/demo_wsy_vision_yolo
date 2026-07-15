#include "timer.h"

void Timer::start()
{
    start_time_ = Clock::now();
    running_ = true;
}

void Timer::stop()
{
    if (!running_)
    {
        return;
    }

    const Clock::time_point end_time =
        Clock::now();

    frame_time_ms_ =
        std::chrono::duration<double, std::milli>(
            end_time - start_time_
        ).count();

    total_time_ms_ += frame_time_ms_;
    ++frame_count_;

    running_ = false;
}

double Timer::getFrameTimeMs() const
{
    return frame_time_ms_;
}

double Timer::getFPS() const
{
    if (frame_time_ms_ <= 0.0)
    {
        return 0.0;
    }

    return 1000.0 / frame_time_ms_;
}

double Timer::getAverageFPS() const
{
    if (frame_count_ <= 0 ||
        total_time_ms_ <= 0.0)
    {
        return 0.0;
    }

    return
        1000.0 *
        static_cast<double>(frame_count_) /
        total_time_ms_;
}

void Timer::reset()
{
    frame_time_ms_ = 0.0;
    total_time_ms_ = 0.0;
    frame_count_ = 0;
    running_ = false;
}