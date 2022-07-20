/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// This code sample demonstrates how to use Metavision Core SDK pipeline utility to create a pipeline displaying the
// results of the sparse optical flow algorithm.

#include <iostream>
#include <functional>
#include <chrono>
#include <boost/program_options.hpp>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/core/pipeline/video_writing_stage.h>
#include <metavision/sdk/cv/pipeline/flow_frame_generation_stage.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/sdk/cv/algorithms/sparse_optical_flow_algorithm.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/ui/pipeline/frame_display_stage.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    std::string bias_file_path;
    uint32_t afk_min_freq = 90;
    uint32_t afk_max_freq = 120;
    bool afk_band_path    = true;
    std::string in_raw_file_path;
    std::string out_avi_file_path;
    bool no_display              = false;
    bool realtime_playback_speed = true;

    const std::string program_desc(
        "Code sample showing how to use Metavision SDK to display results of sparse optical flow.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("bias-file,b", po::value<std::string>(&bias_file_path), "Apply bias settings on the camera")
        ("min-freq,m", po::value<uint32_t>(&afk_min_freq), "AFK: Lowest frequency of the band (in Hz), for Gen4.1 sensors and newer")
        ("max-freq,M", po::value<uint32_t>(&afk_max_freq), "AFK: Highest frequency of the band (in Hz), for Gen4.1 sensors and newer")
        ("band-path,s", po::value<bool>(&afk_band_path), "AFK: True for band-cut (by default), and False for band-pass, for Gen4.1 sensors and newer")
        ("input-raw-file,i", po::value<std::string>(&in_raw_file_path), "Path to input RAW file. If not specified, the camera live stream is used.")
        ("output-avi-file,o", po::value<std::string>(&out_avi_file_path)->default_value("/tmp"), "Path to output AVI file.")
        ("no-display,d", po::bool_switch(&no_display)->default_value(false), "Disable output display window")
        ("realtime-playback-speed", po::value<bool>(&realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible")
        ;
    // clang-format on

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();

    // A pipeline with a main (UI) thread and as many threads as detached stages that will be added
    // Here, all stages that need to be run in their own processing threads are manually detached for conciseness.
    // Alternatively, you can automatically detach all added stages with the Pipeline(true) constructor but it is
    // advised to do so only when you are sure your code is properly functioning since debugging can be problematic
    // when stages are run concurrently.
    Metavision::Pipeline p;

    // Initialize the camera
    Metavision::Camera camera;
    if (in_raw_file_path.empty()) {
        try {
            camera = Metavision::Camera::from_first_available();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
        // Configure biases:
        if (!bias_file_path.empty()) {
            camera.biases().set_from_file(bias_file_path);
        }
        // Configure the AFK (for Gen4.1 sensors and newer)
        if (vm.count("min-freq") || vm.count("max-freq")) {
            auto &module = camera.antiflicker_module();
            module.set_frequency_band(afk_min_freq, afk_max_freq, afk_band_path);
            module.enable();
        }
    } else {
        camera = Metavision::Camera::from_file(in_raw_file_path, realtime_playback_speed);
    }

    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    /// Pipeline
    //
    //  0 (Cam) -->-- 1 (STC) ----------->-----------  2 (Flow)
    //                |                                |
    //                v                                v
    //                |                                |
    //                |------>------->-<--------<------|
    //                                |
    //                                v
    //                                |
    //                                3 (Flow Frame Generator)
    //                                |
    //                                v
    //                                |
    //                |------<-------<->-------->------|
    //                |                                |
    //                v                                v
    //                |                                |
    //                4 (Video writer)                 5 (Display)
    //

    // 0) Stage that will produce events using a Camera instance
    auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(camera)));
    // This stage will run on one of the processing threads
    cam_stage.detach();

    // 1) Stage that will apply STC filtering on the events produced by its previous stage (the camera stage)
    // Note that this stage is created using an instance of one of the provided SDK CV algorithms.
    // Any of the algorithms provided can be added to the pipeline.
    auto &stc_stage = p.add_algorithm_stage(
        std::make_unique<Metavision::SpatioTemporalContrastAlgorithm>(width, height, 0), cam_stage);
    // In the same way, you can access the algo from the stage and use its corresponding functionalities
    stc_stage.algo().set_threshold(40000);
    // This stage will run on one of the processing threads
    stc_stage.detach();

    // 2) Stage that will compute a sparse optical flow given events provided by the previous stage (the STC
    // filtering stage)
    auto &flow_stage = p.add_algorithm_stage<Metavision::EventOpticalFlow>(
        std::make_unique<Metavision::SparseOpticalFlowAlgorithm>(Metavision::SparseOpticalFlowAlgorithm::Parameters(
            width, height, Metavision::SparseOpticalFlowAlgorithm::Parameters::Tuning::FastObjects)),
        stc_stage);
    // This stage will run on one of the processing threads
    flow_stage.detach();

    // 3) Stage that will create a frame given data provided by the previous stages.
    // The class constructor sets customized callbacks for corresponding STC filtering and flow stage.
    auto &frame_stage = p.add_stage(std::make_unique<Metavision::FlowFrameGenerationStage>(width, height, 30));
    frame_stage.set_previous_cd_stage(stc_stage);
    frame_stage.set_previous_flow_stage(flow_stage);
    // As usual, this stage will run on one of the processing threads
    frame_stage.detach();

    // 4) Stage that will create a video from the frames generated by the previous stage
    std::ostringstream video_path;
    video_path << out_avi_file_path << "/output.avi";
    auto &video_stage =
        p.add_stage(std::make_unique<Metavision::VideoWritingStage>(video_path.str(), width, height, 60), frame_stage);
    // This stage will run on one of the processing threads
    video_stage.detach();

    // 5) Stage that will display the frame produced by the previous stage (the frame generation stage).
    if (!no_display) {
        auto &disp_stage = p.add_stage(
            std::make_unique<Metavision::FrameDisplayStage>("Sparse Optical Flow", width, height), frame_stage);
    }
    // Note that since we did not call detach() on this stage, it will run on the main (UI) thread

    p.run();

    const auto end     = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MV_LOG_INFO() << "Ran in" << Metavision::Log::no_space << static_cast<float>(elapsed.count()) / 1000.f << "s";
    MV_LOG_INFO() << "Wrote video file:" << video_path.str();

    return 0;
}
