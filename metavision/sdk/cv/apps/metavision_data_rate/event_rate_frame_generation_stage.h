/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_EVENT_RATE_FRAME_GENERATION_STAGE_H
#define METAVISION_SDK_EVENT_RATE_FRAME_GENERATION_STAGE_H

#include <opencv2/opencv.hpp>

#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/core/algorithms/base_frame_generation_algorithm.h"

#include "event_rate_struct.h"

namespace Metavision {

struct FrameGenerationEventRateStage : public BaseStage {
    using FramePool = SharedObjectPool<cv::Mat>;
    using FramePtr  = FramePool::ptr_type;
    using FrameData = std::pair<timestamp, FramePtr>;

    using EventRateBuffer     = std::vector<EventRateStruct>;
    using EventRateBufferPool = SharedObjectPool<EventRateBuffer>;
    using EventRateBufferPtr  = EventRateBufferPool::ptr_type;

    /// @brief Constructor
    ///
    /// This constructor expects the user to eventually call set_consuming_callback
    /// with the stages producing CD and event rate events.
    ///
    /// @param width Width of the generated frame (in pixels)
    /// @param height Height of the generated frame (in pixels)
    /// @param fps The frame rate of the generated sequence of frames
    FrameGenerationEventRateStage(int width, int height, int fps) :
        width_(width),
        height_(height),
        // using a queue of 2 frames to not pre-compute too many frames in advance
        // which uses a lot of memory, and makes real-time interactions weird (due to the
        // interaction operating on frames which will be displayed only much later)
        frame_pool_(FramePool::make_bounded(2)) {
        display_accumulation_time_us_ = 10000;
        bg_color_                     = BaseFrameGenerationAlgorithm::get_cv_color(Metavision::ColorPalette::Light,
                                                               Metavision::ColorType::Background);
        on_color_                     = BaseFrameGenerationAlgorithm::get_cv_color(Metavision::ColorPalette::Light,
                                                               Metavision::ColorType::Positive);
        off_color_                    = BaseFrameGenerationAlgorithm::get_cv_color(Metavision::ColorPalette::Light,
                                                                Metavision::ColorType::Negative);
        text_color_                   = cv::Vec3b(52, 37, 30);
        text_position_                = cv::Point(20, 40);
        colored_                      = true;
        frame_period_                 = static_cast<timestamp>(1.e6 / fps + 0.5);
        next_process_ts_              = frame_period_;
        ts_history_                   = cv::Mat_<std::int32_t>(height, width, -1);
        set_consuming_callback([this](const boost::any &data) {
            MV_SDK_LOG_WARNING() << "For this stage to work properly, you need to call set_previous_cd_stage\n"
                                 << "and set_previous_er_stage with the stages producing CD, resp. event rate events.";
        });
    }

    /// @brief Constructor
    ///
    /// Convenience overload that directly sets the consuming callback
    /// @param width Width of the generated frame
    /// @param height Height of the generated frame
    /// @param fps The frame rate of the generated sequence of frames
    /// @param prev_cd_stage the stage producing CD events to be consumed
    /// @param prev_er_stage the stage producing event rate events to be consumed
    FrameGenerationEventRateStage(BaseStage &prev_cd_stage, BaseStage &prev_er_stage, int width, int height, int fps) :
        FrameGenerationEventRateStage(width, height, fps) {
        set_previous_cd_stage(prev_cd_stage);
        set_previous_er_stage(prev_er_stage);
    }

    /// @brief Sets the previous cd stage producing CD events
    /// @param prev_cd_stage Stage producing CD events to be consumed
    void set_previous_cd_stage(BaseStage &prev_cd_stage) {
        set_consuming_callback(prev_cd_stage, [this](const boost::any &data) { consume_cd_events(data); });
    }

    /// @brief Sets the previous event rate stage producing event rate events
    /// @param prev_stage Stage producing event rate events to be consumed
    void set_previous_er_stage(BaseStage &prev_stage) {
        set_consuming_callback(prev_stage, [this](const boost::any &data) { consume_er_events(data); });
    }

private:
    void consume_cd_events(const boost::any &data) {
        try {
            auto buffer = boost::any_cast<EventBufferPtr>(data);
            if (!buffer->empty()) {
                // duplicate the buffer to avoid depleting the previous stages' object pool
                auto out_buffer = event_buffer_pool_.acquire();
                *out_buffer     = *buffer;

                cd_buffers_.emplace(out_buffer);
                timestamp last_cd_ts = -1, last_er_ts = -1;
                if (!cd_buffers_.empty() && !cd_buffers_.back()->empty())
                    last_cd_ts = cd_buffers_.back()->back().t;
                if (!er_buffers_.empty() && !er_buffers_.back()->empty())
                    last_er_ts = er_buffers_.back()->back().t;
                while (last_cd_ts >= next_process_ts_ && last_er_ts >= next_process_ts_) {
                    generate();
                }
            }
        } catch (boost::bad_any_cast &) {}
    }

    void consume_er_events(const boost::any &data) {
        try {
            auto buffer = boost::any_cast<EventRateBufferPtr>(data);
            if (!buffer->empty()) {
                er_buffers_.emplace(buffer);
                timestamp last_cd_ts = -1, last_er_ts = -1;
                if (!cd_buffers_.empty() && !cd_buffers_.back()->empty())
                    last_cd_ts = cd_buffers_.back()->back().t;
                if (!er_buffers_.empty() && !er_buffers_.back()->empty())
                    last_er_ts = er_buffers_.back()->back().t;
                while (last_cd_ts >= next_process_ts_ && last_er_ts >= next_process_ts_) {
                    generate();
                }
            }
        } catch (boost::bad_any_cast &) {}
    }

    void generate() {
        auto cur_frame     = frame_pool_.acquire();
        timestamp ts_end   = next_process_ts_;
        timestamp ts_begin = ts_end - display_accumulation_time_us_;
        update_frame_with_cd(ts_begin, ts_end, *cur_frame);
        update_frame_with_er(ts_begin, ts_end, *cur_frame);
        produce(std::make_pair(next_process_ts_, cur_frame));
        next_process_ts_ += frame_period_;
    }

    void update_frame_with_cd(timestamp ts_begin, timestamp ts_end, cv::Mat &frame) {
        // update frame type if needed
        frame.create(height_, width_, colored_ ? CV_8UC3 : CV_8U);

        // if the timestamps we want to store overflow a 32 bits integer value
        // we increase the offset and update the stored timestamps
        while (ts_end > ts_offset_ + std::numeric_limits<std::int32_t>::max()) {
            ts_offset_ += std::numeric_limits<std::int32_t>::max();
            ts_history_ -= std::numeric_limits<std::int32_t>::max();
        }

        while (!cd_buffers_.empty()) {
            auto buffer = cd_buffers_.front();
            if (!buffer->empty()) {
                auto it_begin = std::lower_bound(std::begin(*buffer), std::end(*buffer), EventCD(0, 0, 0, ts_begin),
                                                 [](const auto &ev1, const auto &ev2) { return ev1.t < ev2.t; });
                auto it_end   = std::lower_bound(std::begin(*buffer), std::end(*buffer), EventCD(0, 0, 0, ts_end),
                                               [](const auto &ev1, const auto &ev2) { return ev1.t < ev2.t; });
                if (it_begin != it_end) {
                    if (colored_) {
                        for (auto it = it_begin; it != it_end; ++it) {
                            ts_history_(it->y, it->x)         = static_cast<std::int32_t>(it->t - ts_offset_);
                            frame.at<cv::Vec3b>(it->y, it->x) = it->p ? on_color_ : off_color_;
                        }
                    } else {
                        for (auto it = it_begin; it != it_end; ++it) {
                            ts_history_(it->y, it->x)       = static_cast<std::int32_t>(it->t - ts_offset_);
                            frame.at<uint8_t>(it->y, it->x) = it->p ? on_color_[0] : off_color_[0];
                        }
                    }
                }
                if (it_end != std::end(*buffer))
                    break;
            }
            cd_buffers_.pop();
        }

        // reset pixels for too old events
        frame.setTo(bg_color_, ts_history_ < static_cast<std::int32_t>(ts_begin - ts_offset_));
    }

    void update_frame_with_er(timestamp ts_begin, timestamp ts_end, cv::Mat &frame) {
        if (er_buffers_.empty())
            return;

        int event_rate_max = 0;
        while (!er_buffers_.empty()) {
            auto buffer = er_buffers_.front();
            if (!buffer->empty()) {
                if (event_rate_max < buffer->front().rate)
                    event_rate_max = buffer->front().rate;
            }
            er_buffers_.pop();
        }

        std::stringstream ss_er;
        if (event_rate_max >= 4)
            ss_er << "Event rate : " << event_rate_max << "kEv/s";
        else
            ss_er << "Event rate : 0kEv/s";
        cv::putText(frame, ss_er.str(), text_position_, cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color_, 1, cv::LINE_8);
    }

    // Vector of timestamps
    cv::Mat_<std::int32_t> ts_history_;
    timestamp ts_offset_ = 0;

    // Image to display
    int width_, height_;
    timestamp frame_period_ = -1;
    cv::Vec3b bg_color_, on_color_, off_color_, text_color_;
    bool colored_;
    cv::Point text_position_;

    // Time interval to display events
    uint32_t display_accumulation_time_us_ = 5000;

    // Next processing timestamp according to frame_period value
    timestamp next_process_ts_ = 0;

    SharedObjectPool<EventBuffer> event_buffer_pool_;
    std::queue<SharedObjectPool<EventBuffer>::ptr_type> cd_buffers_;
    std::queue<EventRateBufferPtr> er_buffers_;

    FramePool frame_pool_;
};

} // namespace Metavision

#endif // METAVISION_SDK_EVENT_RATE_FRAME_GENERATION_STAGE_H
