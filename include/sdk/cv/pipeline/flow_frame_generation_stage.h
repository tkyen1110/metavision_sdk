/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_FLOW_FRAME_GENERATION_STAGE_H
#define METAVISION_SDK_CV_FLOW_FRAME_GENERATION_STAGE_H

#include <opencv2/opencv.hpp>

#include "metavision/sdk/core/pipeline/base_stage.h"
#include "metavision/sdk/base/utils/object_pool.h"
#include "metavision/sdk/cv/algorithms/flow_frame_generator_algorithm.h"

namespace Metavision {

struct FlowFrameGenerationStage : public BaseStage {
    using FramePool = SharedObjectPool<cv::Mat>;
    using FramePtr  = FramePool::ptr_type;
    using FrameData = std::pair<timestamp, FramePtr>;

    using EventFlowBuffer     = std::vector<EventOpticalFlow>;
    using EventFlowBufferPool = SharedObjectPool<EventFlowBuffer>;
    using EventFlowBufferPtr  = EventFlowBufferPool::ptr_type;

    /// @brief Constructor
    ///
    /// This constructor expects the user to eventually call set_consuming_callback
    /// with the stages producing CD and flow events.
    ///
    /// @param width Width of the generated frame (in pixels)
    /// @param height Height of the generated frame (in pixels)
    /// @param fps The frame rate of the generated sequence of frames
    FlowFrameGenerationStage(int width, int height, int fps) :
        width_(width),
        height_(height),
        // using a queue of 2 frames to not pre-compute too many frames in advance
        // which uses a lot of memory, and makes real-time interactions weird (due to the
        // interaction operating on frames which will be displayed only much later)
        frame_pool_(FramePool::make_bounded(2)) {
        display_accumulation_time_us_ = 10000;
        bg_color_                     = cv::Vec3b(52, 37, 30);
        on_color_                     = cv::Vec3b(236, 223, 216);
        off_color_                    = cv::Vec3b(201, 126, 64);
        colored_                      = true;
        frame_period_                 = static_cast<timestamp>(1.e6 / fps + 0.5);
        next_process_ts_              = frame_period_;
        cd_frame_updated_             = false;
        set_consuming_callback([this](const boost::any &data) {
            MV_SDK_LOG_WARNING() << "For this stage to work properly, you need to call set_previous_cd_stage\n"
                                 << "and set_previous_flow_stage with the stages producing CD, resp. flow events.";
        });
    }

    /// @brief Constructor
    ///
    /// Convenience overload that directly sets the consuming callback
    /// @param width Width of the generated frame
    /// @param height Height of the generated frame
    /// @param fps The frame rate of the generated sequence of frames
    /// @param prev_cd_stage the stage producing CD events to be consumed
    /// @param prev_flow_stage the stage producing flow events to be consumed
    FlowFrameGenerationStage(BaseStage &prev_cd_stage, BaseStage &prev_flow_stage, int width, int height, int fps) :
        FlowFrameGenerationStage(width, height, fps) {
        set_previous_cd_stage(prev_cd_stage);
        set_previous_flow_stage(prev_flow_stage);
    }

    /// @brief Sets the previous cd stage producing CD events
    /// @param prev_cd_stage Stage producing CD events to be consumed
    void set_previous_cd_stage(BaseStage &prev_cd_stage) {
        set_consuming_callback(prev_cd_stage, [this](const boost::any &data) { consume_cd_events(data); });
    }

    /// @brief Sets the previous flow stage producing flow events
    /// @param prev_flow_stage Stage producing flow events to be consumed
    void set_previous_flow_stage(BaseStage &prev_flow_stage) {
        set_consuming_callback(prev_flow_stage, [this](const boost::any &data) { consume_flow_events(data); });
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
                timestamp last_cd_ts = -1, last_flow_ts = -1;
                if (!cd_buffers_.empty() && !cd_buffers_.back()->empty())
                    last_cd_ts = cd_buffers_.back()->back().t;
                if (!flow_buffers_.empty() && !flow_buffers_.back()->empty())
                    last_flow_ts = flow_buffers_.back()->back().t;
                while (last_cd_ts >= next_process_ts_ && last_flow_ts >= next_process_ts_) {
                    generate();
                }
            }
        } catch (boost::bad_any_cast &) {}
    }

    void consume_flow_events(const boost::any &data) {
        try {
            auto buffer = boost::any_cast<EventFlowBufferPtr>(data);
            if (!buffer->empty()) {
                flow_buffers_.emplace(buffer);
                timestamp last_cd_ts = -1, last_flow_ts = -1;
                if (!cd_buffers_.empty() && !cd_buffers_.back()->empty())
                    last_cd_ts = cd_buffers_.back()->back().t;
                if (!flow_buffers_.empty() && !flow_buffers_.back()->empty())
                    last_flow_ts = flow_buffers_.back()->back().t;
                while (last_cd_ts >= next_process_ts_ && last_flow_ts >= next_process_ts_) {
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
        update_frame_with_flow(ts_begin, ts_end, *cur_frame);
        if (cd_frame_updated_) {
            // ignore flow if cd frame has not been updated
            produce(std::make_pair(next_process_ts_, cur_frame));
            cd_frame_updated_ = false;
        }
        next_process_ts_ += frame_period_;
    }

    void update_frame_with_cd(timestamp ts_begin, timestamp ts_end, cv::Mat &frame) {
        // update frame type if needed
        frame.create(height_, width_, colored_ ? CV_8UC3 : CV_8U);
        frame.setTo(bg_color_);

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
                            frame.at<cv::Vec3b>(it->y, it->x) = it->p ? on_color_ : off_color_;
                        }
                    } else {
                        for (auto it = it_begin; it != it_end; ++it) {
                            frame.at<uint8_t>(it->y, it->x) = it->p ? on_color_[0] : off_color_[0];
                        }
                    }
                    cd_frame_updated_ = true;
                }
                if (it_end != std::end(*buffer))
                    break;
            }
            cd_buffers_.pop();
        }
    }

    void update_frame_with_flow(timestamp ts_begin, timestamp ts_end, cv::Mat &frame) {
        while (!flow_buffers_.empty()) {
            auto buffer = flow_buffers_.front();
            if (!buffer->empty()) {
                auto it_begin = std::lower_bound(std::begin(*buffer), std::end(*buffer), ts_begin,
                                                 [](const auto &ev, timestamp t) { return ev.t < t; });
                auto it_end   = std::lower_bound(std::begin(*buffer), std::end(*buffer), ts_end,
                                               [](const auto &ev, timestamp t) { return ev.t < t; });
                if (it_begin != it_end) {
                    algo_.add_flow_for_frame_update(it_begin, it_end);
                }
                if (it_end != std::end(*buffer))
                    break;
            }
            flow_buffers_.pop();
        }
        algo_.clear_ids();
        algo_.update_frame_with_flow(frame);
    }

    // Image to display
    int width_, height_;
    timestamp frame_period_ = -1;
    cv::Scalar bg_color_;
    cv::Vec3b on_color_, off_color_;
    bool colored_;
    bool cd_frame_updated_;

    // Time interval to display events
    uint32_t display_accumulation_time_us_ = 5000;

    // Next processing timestamp according to frame_period value
    timestamp next_process_ts_ = 0;

    SharedObjectPool<EventBuffer> event_buffer_pool_;
    std::queue<SharedObjectPool<EventBuffer>::ptr_type> cd_buffers_;
    std::queue<EventFlowBufferPtr> flow_buffers_;
    FlowFrameGeneratorAlgorithm algo_;

    FramePool frame_pool_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_FLOW_FRAME_GENERATION_STAGE_H
