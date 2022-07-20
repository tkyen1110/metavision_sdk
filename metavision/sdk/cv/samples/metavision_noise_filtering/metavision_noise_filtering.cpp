/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to use Metavision SDK CV module to test different noise filtering strategies.
// In addition, it shows how to capture the keys pressed in a display window so as to modify the behavior of the stages
// while the pipeline is running.

#include <functional>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/frame_generation_stage.h>
#include <metavision/sdk/cv/algorithms/activity_noise_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/trail_filter_algorithm.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string in_raw_file_path;
    std::string active_filter;
    bool no_display              = false;
    bool realtime_playback_speed = true;

    const std::string short_program_desc("Code sample showing how the pipeline framework can be used to "
                                         "create a simple application testing different noise filtering strategies.\n");

    const std::string long_program_desc(short_program_desc +
                                        "Available keyboard options:\n"
                                        "  - a - filter events using the activity noise filter algorithm\n"
                                        "  - t - filter events using the trail filter algorithm\n"
                                        "  - s - filter events using the spatio temporal contrast algorithm\n"
                                        "  - e - show all events\n"
                                        "  - q - quit the application\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i", po::value<std::string>(&in_raw_file_path), "Path to input RAW file. If not specified, the camera live stream is used.")
        ("activate-filter,f", po::value<std::string>(&active_filter), "Filter to activate by default : [activity/trail/spatio]")
        ("no-display,d", po::bool_switch(&no_display)->default_value(false), "Disable output display window")
        ("realtime-playback-speed", po::value<bool>(&realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible")
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

    MV_LOG_INFO() << long_program_desc;

    // A pipeline for which all added stages will automatically be run in their own processing threads (if applicable)
    Metavision::Pipeline p(true);

    // Construct a camera from a recording or a live stream
    Metavision::Camera cam;
    if (in_raw_file_path.size()) {
        cam = Metavision::Camera::from_file(in_raw_file_path, realtime_playback_speed);
    } else {
        cam = Metavision::Camera::from_first_available();
    }
    const unsigned short width  = cam.geometry().width();
    const unsigned short height = cam.geometry().height();

    // Pipeline
    //                  +--->  1a (Activity Noise filter) +---v
    //                  |                                     |
    //                  |                                     |
    // 0 (Camera)    ---------->  1b (Trail filter)  +------------->  2 (Frame generation)  +------>   3 (Display)
    //                  |                                     |
    //                  |                                     |
    //                  +------>   1c (STC filter)   +--------^

    // 0) Stage producing events from a camera
    auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(cam)));

    // 1a) Stage wrapping an activity filter algorithm
    auto &activity_filter_stage =
        p.add_algorithm_stage(std::make_unique<Metavision::ActivityNoiseFilterAlgorithm<>>(width, height, 20000),
                              cam_stage, active_filter == "activity");

    // 1b) Stage wrapping a trail filter algorithm
    auto &trail_filter_stage =
        p.add_algorithm_stage(std::make_unique<Metavision::TrailFilterAlgorithm>(width, height, 1000000),
                              activity_filter_stage, active_filter == "trail");

    // 1c) Stage wrapping a spatio temporal contrast filter
    auto &stc_filter_stage =
        p.add_algorithm_stage(std::make_unique<Metavision::SpatioTemporalContrastAlgorithm>(width, height, 10000, true),
                              trail_filter_stage, active_filter == "spatio");

    // 2) Stage generating a frame from filtered events using accumulation time of 30ms
    auto &frame_stage =
        p.add_stage(std::make_unique<Metavision::FrameGenerationStage>(width, height, 30), stc_filter_stage);

    // 3) Stage displaying the frame
    auto key_callback_fn = [&](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
        if (action == Metavision::UIAction::RELEASE) {
            switch (key) {
            case Metavision::UIKeyEvent::KEY_A:
                // enable the activity filter
                MV_LOG_INFO() << "Activity filter enabled";

                activity_filter_stage.set_enabled(true);
                trail_filter_stage.set_enabled(false);
                stc_filter_stage.set_enabled(false);
                break;
            case Metavision::UIKeyEvent::KEY_T:
                // enable the trail filter
                MV_LOG_INFO() << "Trail filter enabled";

                activity_filter_stage.set_enabled(false);
                trail_filter_stage.set_enabled(true);
                stc_filter_stage.set_enabled(false);
                break;
            case Metavision::UIKeyEvent::KEY_S:
                // enable the spatio temporal contrast filter
                MV_LOG_INFO() << "Spatio temporal contrast filter enabled";

                activity_filter_stage.set_enabled(false);
                trail_filter_stage.set_enabled(false);
                stc_filter_stage.set_enabled(true);
                break;
            case Metavision::UIKeyEvent::KEY_E:
                // show all events (no filtering enabled)
                MV_LOG_INFO() << "Noise filtering disabled";

                activity_filter_stage.set_enabled(false);
                trail_filter_stage.set_enabled(false);
                stc_filter_stage.set_enabled(false);
                break;
            }
        }
    };
    if (!no_display) {
        auto &disp_stage =
            p.add_stage(std::make_unique<Metavision::FrameDisplayStage>("CD events", width, height), frame_stage);
        disp_stage.set_key_callback(key_callback_fn);
    }

    // Run the pipeline
    p.run();

    return 0;
}
