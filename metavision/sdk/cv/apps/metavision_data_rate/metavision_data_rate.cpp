/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This application applies noise filtering and computes the event rate afterwards.

#include <functional>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/cv/algorithms/anti_flicker_algorithm.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>

#include "event_rate_algorithm.h"
#include "event_rate_frame_generation_stage.h"

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_raw_file_path;
    std::string bias_file_path;
    uint32_t afk_min_freq = 90;
    uint32_t afk_max_freq = 120;
    bool afk_band_path    = true;

    int act_filter_th = 10000;
    int stc_filter_th = 10000;

    int do_activity = 1;
    bool do_stc     = false;
    bool do_af      = false;

    const std::string short_program_desc("Data Rate Demo\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")

        ("bias-file,b", po::value<std::string>(&bias_file_path), "Apply bias settings on the camera")
        ("min-freq,m", po::value<uint32_t>(&afk_min_freq), "AFK: Lowest frequency of the band (in Hz), for Gen4.1 sensors and newer")
        ("max-freq,M", po::value<uint32_t>(&afk_max_freq), "AFK: Highest frequency of the band (in Hz), for Gen4.1 sensors and newer")
        ("band-path,s", po::value<bool>(&afk_band_path), "AFK: True for band-cut (by default), and False for band-pass, for Gen4.1 sensors and newer")

        ("apply-activity-filter", po::value<int>(&do_activity)->default_value(1), "Apply Activity Filter (on by default)")
        ("activity-th", po::value<int>(&act_filter_th)->default_value(10000), "Activity Filter Threshold")
        ("apply-stc-filter", po::bool_switch(&do_stc), "Apply STC filter (off by default)")
        ("stc-th", po::value<int>(&stc_filter_th)->default_value(10000), "STC Filter Threshold")
        ("apply-anti-flicker-filter", po::bool_switch(&do_af), "Apply anti-flicker filter (off by default)")
        ("input-raw-file,i", po::value<std::string>(&in_raw_file_path), "Path to input RAW file. If not specified, the camera live stream is used.")
        ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << short_program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << short_program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    // A pipeline for which all added stages will automatically be run in their own processing threads (if applicable)
    Metavision::Pipeline p(true);

    // Construct a camera from a recording or a live stream
    Metavision::Camera cam;
    if (!in_raw_file_path.empty()) {
        cam = Metavision::Camera::from_file(in_raw_file_path);
    } else {
        cam = Metavision::Camera::from_first_available();

        // Configure biases:
        if (!bias_file_path.empty()) {
            cam.biases().set_from_file(bias_file_path);
        }
        // Configure the AFK (for Gen4.1 sensors and newer)
        if (vm.count("min-freq") || vm.count("max-freq")) {
            auto &module = cam.antiflicker_module();
            module.set_frequency_band(afk_min_freq, afk_max_freq, afk_band_path);
            module.enable();
        }
    }

    const unsigned short width  = cam.geometry().width();
    const unsigned short height = cam.geometry().height();

    // 0) Stage producing events from a camera
    auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(cam)));

    // 1a) Stage wrapping an activity filter algorithm
    auto &activity_filter_stage = p.add_algorithm_stage(
        std::make_unique<Metavision::ActivityNoiseFilterAlgorithm<>>(width, height, act_filter_th), cam_stage, false);
    activity_filter_stage.set_enabled(do_activity);

    // 1b) Stage wrapping a spatio temporal contrast filter
    auto &stc_filter_stage = p.add_algorithm_stage(
        std::make_unique<Metavision::SpatioTemporalContrastAlgorithm>(width, height, stc_filter_th, true),
        activity_filter_stage, false);
    stc_filter_stage.set_enabled(do_stc);
    stc_filter_stage.detach();

    // 1c) anti-flicker filter
    Metavision::FrequencyEstimationConfig anti_flicker_config(7, 95, 125);
    auto &anti_flicker_stage =
        p.add_algorithm_stage(std::make_unique<Metavision::AntiFlickerAlgorithm>(width, height, anti_flicker_config),
                              stc_filter_stage, false);
    anti_flicker_stage.set_enabled(do_af);
    anti_flicker_stage.detach();

    // 2) Event rate
    auto &event_rate_stage = p.add_algorithm_stage<Metavision::EventRateStruct>(
        std::make_unique<Metavision::EventRateAlgorithm>(), anti_flicker_stage);
    event_rate_stage.detach();

    // 3) Stage generating a frame from filtered events
    auto &frame_stage =
        p.add_stage(std::make_unique<Metavision::FrameGenerationEventRateStage>(width, height, 30), anti_flicker_stage);
    frame_stage.set_previous_cd_stage(anti_flicker_stage);
    frame_stage.set_previous_er_stage(event_rate_stage);
    frame_stage.detach();

    // 4) Stage displaying the frame
    auto &disp_stage =
        p.add_stage(std::make_unique<Metavision::FrameDisplayStage>("CD events", width, height), frame_stage);

    // Run the pipeline
    p.run();

    return 0;
}
