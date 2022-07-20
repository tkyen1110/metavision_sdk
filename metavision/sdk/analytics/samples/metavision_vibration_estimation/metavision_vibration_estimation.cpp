/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

// Example of using Metavision SDK Analytics API to estimate vibration

#include <iostream>
#include <functional>
#include <chrono>
#include <boost/program_options.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/base/utils/object_pool.h>
#include <metavision/sdk/core/pipeline/pipeline.h>
#include <metavision/sdk/driver/pipeline/camera_stage.h>
#include <metavision/sdk/cv/configs/frequency_estimation_config.h>
#include <metavision/sdk/analytics/algorithms/frequency_map_async_algorithm.h>
#include <metavision/sdk/analytics/algorithms/heat_map_frame_generator_algorithm.h>
#include <metavision/sdk/analytics/algorithms/dominant_value_map_algorithm.h>
#include <metavision/sdk/ui/utils/mt_window.h>
#include <metavision/sdk/ui/utils/event_loop.h>

namespace po = boost::program_options;

using EventBuffer    = std::vector<Metavision::EventCD>;
using EventBufferPtr = Metavision::SharedObjectPool<EventBuffer>::ptr_type;

using FrequencyMap      = Metavision::FrequencyMapAsyncAlgorithm::OutputMap;
using FrequencyMapPool  = Metavision::SharedObjectPool<FrequencyMap>;
using FrequencyMapPtr   = FrequencyMapPool::ptr_type;
using TimedFrequencyMap = std::pair<Metavision::timestamp, FrequencyMapPtr>;

/// @brief Stage that estimates the pixel-wise frequency of vibrating objects to produce a frequency map.
///
/// The stage takes as:
///   - Input : buffer of events                        : EventBufferPtr
///   - Output: timestamped float frame (Frequency Map) : TimedFrequencyMap
class FrequencyMapAsyncAlgorithmStage : public Metavision::BaseStage {
public:
    /// @brief Constructs a new Frequency Map Async Algorithm stage
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param freq_config Configuration of the @ref Metavision::FrequencyMapAsyncAlgorithm
    /// @param update_frequency Frequency at which a new frequency map is produced
    FrequencyMapAsyncAlgorithmStage(int width, int height, const Metavision::FrequencyEstimationConfig &freq_config,
                                    float update_frequency) {
        freq_algo_ = std::make_unique<Metavision::FrequencyMapAsyncAlgorithm>(width, height, freq_config);
        freq_algo_->set_update_frequency(update_frequency);

        /// [SET_CD_EVENTS_CONSUMING_CALLBACK_BEGIN]
        set_consuming_callback([this](const boost::any &data) {
            try {
                auto buffer = boost::any_cast<EventBufferPtr>(data);
                if (buffer->empty())
                    return;

                freq_algo_->process_events(buffer->cbegin(), buffer->cend());

            } catch (boost::bad_any_cast &c) { MV_LOG_ERROR() << c.what(); }
        });
        /// [SET_CD_EVENTS_CONSUMING_CALLBACK_END]

        /// [SET_FREQUENCY_MAP_OUTPUT_CALLBACK_BEGIN]
        frequency_map_pool_ = FrequencyMapPool::make_bounded();
        freq_algo_->set_output_callback([this](Metavision::timestamp t, FrequencyMap &frequency_map) {
            TimedFrequencyMap timed_frequency_map;
            timed_frequency_map.first  = t;
            timed_frequency_map.second = frequency_map_pool_.acquire();

            // The frequency map is passed via a non constant reference meaning that we are free to swap it to avoid
            // useless copies.
            cv::swap(*timed_frequency_map.second, frequency_map);

            produce(timed_frequency_map);
        });
        /// [SET_FREQUENCY_MAP_OUTPUT_CALLBACK_END]
    }

private:
    std::unique_ptr<Metavision::FrequencyMapAsyncAlgorithm> freq_algo_;
    FrequencyMapPool frequency_map_pool_;
};

/// @brief GUI stage
///
/// Displays every received frequency map using a color map and displays the dominant frequency. In addition, this GUI
/// stage allows:
///   - defining ROIs in the frequency map and displaying their dominant frequency,
///   - checking the frequency at a specific position in the frequency map
///
/// The stage takes as:
///   - Input : timestamped float frame (Frequency Map) : TimedFrequencyMap
class VibrationGUIStage : public Metavision::BaseStage {
public:
    /// @brief Constructs a new Vibration GUI stage
    /// @param width Sensor's width
    /// @param height Sensor's height
    /// @param min_freq_hz Minimum detected frequency
    /// @param max_freq_hz Maximum detected frequency
    /// @param freq_precision_hz Precision of the computed dominant frequency
    /// @param min_pixel_count Threshold used in the dominant frequency computation
    VibrationGUIStage(int width, int height, float min_freq_hz, float max_freq_hz, float freq_precision_hz,
                      int min_pixel_count) :
        width_(width), height_(height) {
        freq_map_display_algo_ = std::make_unique<Metavision::HeatMapFrameGeneratorAlgorithm>(
            min_freq_hz, max_freq_hz, freq_precision_hz, width, height, "Hz");
        dominant_freq_algo_ = std::make_unique<Metavision::DominantValueMapAlgorithm<frequency_precision_type>>(
            min_freq_hz, max_freq_hz, freq_precision_hz, min_pixel_count);

        window_ = std::make_unique<Metavision::MTWindow>(WINDOW_NAME, width, freq_map_display_algo_->get_full_height(),
                                                         Metavision::BaseWindow::RenderMode::BGR);

        // clang-format off
        help_msg_ = {"Keyboard/mouse actions:",
                     "  \"h\" - show/hide the help menu",
                     "  \"q\" - quit the application",
                     "  \"c\" - clear all the ROIs",
                     "  Click and drag to create ROIs"};
        // clang-format on

        init_text_rendering();
        print_init_message();

        set_window_callbacks();

        /// [VIBRATION_GUI_STAGE_SET_CONSUMING_CALLBACK_BEGIN]
        set_consuming_callback([this](const boost::any &data) { display_callback(data); });
        /// [VIBRATION_GUI_STAGE_SET_CONSUMING_CALLBACK_END]
    }

private:
    /// @brief Type used to define the type of the boundaries of the histogram bins in the dominant value estimation
    using frequency_precision_type = float;

    /// @brief
    void set_window_callbacks() {
        window_->set_cursor_pos_callback([this](double x, double y) {
            int window_width, window_height;
            window_->get_size(window_width, window_height);

            // The window may have been resized meanwhile. So we map the coordinates to the original window's size.
            const auto mapped_x = static_cast<int>(x * width_ / window_width);
            const auto mapped_y = static_cast<int>(y * freq_map_display_algo_->get_full_height() / window_height);
            cv::Point mouse_pos(mapped_x, mapped_y);

            // The frequency map is smaller than the displayed image (a color map bar is added). So we need to check
            // that we won't be out of bounds.
            cv::Rect img_roi(0, 0, width_, height_);
            if (!img_roi.contains(mouse_pos))
                return;

            // Update the last created ROI
            if (is_updating_roi_ && !rois_.empty()) {
                auto &roi                 = rois_.back();
                const auto &roi_start_pos = crt_roi_start_pos_;

                const auto xmin = std::min(roi_start_pos.x, mouse_pos.x);
                const auto xmax = std::max(roi_start_pos.x, mouse_pos.x);
                const auto ymin = std::min(roi_start_pos.y, mouse_pos.y);
                const auto ymax = std::max(roi_start_pos.y, mouse_pos.y);

                roi = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
            }

            last_mouse_pos_ = mouse_pos;
        });

        window_->set_mouse_callback([this](Metavision::UIMouseButton button, Metavision::UIAction action, int mods) {
            switch (action) {
            case Metavision::UIAction::PRESS:
                // Start a new ROI
                is_updating_roi_   = true;
                crt_roi_start_pos_ = cv::Point(last_mouse_pos_.x, last_mouse_pos_.y);
                rois_.emplace_back(cv::Rect(last_mouse_pos_.x, last_mouse_pos_.y, 1, 1));
                break;

            case Metavision::UIAction::RELEASE:
                // Stop updating the ROI
                is_updating_roi_ = false;
                break;
            }
        });

        window_->set_keyboard_callback(
            [this](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
                if (action == Metavision::UIAction::RELEASE) {
                    switch (key) {
                    case Metavision::UIKeyEvent::KEY_ESCAPE:
                    case Metavision::UIKeyEvent::KEY_Q:
                        this->pipeline().cancel();
                        break;

                    case Metavision::UIKeyEvent::KEY_C:
                        rois_.clear();
                        break;

                    case Metavision::UIKeyEvent::KEY_H:
                        display_help_ = !display_help_;
                        break;

                    default:
                        break;
                    }
                }
            });
    }

    /// @brief Callback called every time a frequency map is received
    ///
    ///    - converts the frequency map to an RGB image,
    ///    - prints the dominant frequency in the whole FOV,
    ///    - prints all the created ROIs their associated dominant frequencies,
    ///    - prints the dominant frequency pointed by the mouse cursor,
    ///    - prints a help message, and
    ///    - displays the generated frame
    /// @param data The input frequency map
    void display_callback(const boost::any &data) {
        try {
            /// [VIBRATION_GUI_DISPLAY_CALLBACK_BEGIN]
            window_->poll_events();

            auto ts_frequency_map = boost::any_cast<TimedFrequencyMap>(data);

            const auto &input_ts     = ts_frequency_map.first;
            auto &input_freq_map_ptr = ts_frequency_map.second;
            if (!input_freq_map_ptr)
                return;

            // Generate the heat map
            freq_map_display_algo_->generate_bgr_heat_map(*input_freq_map_ptr, display_frame_);

            print_dominant_frequency(*input_freq_map_ptr);
            print_pixel_frequency(*input_freq_map_ptr);
            print_rois(*input_freq_map_ptr);
            print_help_message();

            window_->show_async(display_frame_, false);
            if (window_->should_close())
                this->pipeline().cancel();

            /// [VIBRATION_GUI_DISPLAY_CALLBACK_END]
        } catch (boost::bad_any_cast &c) { MV_LOG_ERROR() << c.what(); }
    }

    /// @brief Prints and displays a message in the window
    ///
    /// This is used at the beginning of the application when no frequency map has been received yet.
    void print_init_message() {
        display_frame_.create(height_, width_, CV_8UC3);
        display_frame_.setTo(0);

        const auto font       = cv::FONT_HERSHEY_SIMPLEX;
        const auto font_scale = 1;
        const auto thickness  = 2;
        int base_line         = 0;
        cv::Size text_size;

        const int y_mid = display_frame_.rows / 2;

        const auto print_text = [&](const std::string &text, int y_offset) {
            cv::Point text_org((display_frame_.cols - text_size.width) / 2, y_mid + y_offset);
            cv::putText(display_frame_, text, text_org, font, font_scale, cv::Scalar::all(255), thickness, 8);
        };

        const std::string init_message_1 = "NO VIBRATING OBJECT IN THE";
        const std::string init_message_2 = "FIELD OF VIEW OF THE CAMERA";

        text_size = cv::getTextSize(init_message_1, font, font_scale, thickness, &base_line);
        print_text(init_message_1, -base_line);

        text_size = cv::getTextSize(init_message_2, font, font_scale, thickness, &base_line);
        print_text(init_message_2, text_size.height);

        window_->show_async(display_frame_);
    }

    /// @brief Initializes some parameters used when rendering texts
    void init_text_rendering() {
        const std::string max_dominant_frequency_text = "Frequency: XXXX Hz";
        int base_line                                 = 0;
        const auto text_size =
            cv::getTextSize(max_dominant_frequency_text, FONT_FACE, FONT_SCALE, THICKNESS, &base_line);

        dominant_frequency_text_pos_ = cv::Point(MARGIN, height_ - base_line - MARGIN);
        help_msg_text_pos_           = cv::Point(MARGIN, text_size.height + MARGIN);
        text_height_                 = text_size.height;
    }

    /// @brief Computes and prints the dominant frequency measured in the camera's FOV
    /// @param frequency_map The input frequency map
    void print_dominant_frequency(const FrequencyMap &frequency_map) {
        frequency_precision_type dominant_freq;
        std::ostringstream oss;
        if (dominant_freq_algo_->compute_dominant_value(frequency_map, dominant_freq))
            oss << "Frequency: " << std::setw(4) << dominant_freq << " Hz";
        else
            oss << "Frequency:     N/A";

        cv::putText(display_frame_, oss.str(), dominant_frequency_text_pos_, FONT_FACE, FONT_SCALE,
                    cv::Scalar::all(255), THICKNESS);
    }

    /// @brief Prints the frequency pointed by the mouse cursor
    ///
    /// This is only done if the user is not creating an ROI.
    /// @param frequency_map
    void print_pixel_frequency(const FrequencyMap &frequency_map) {
        const cv::Rect img_roi(0, 0, width_, height_);

        if (!is_updating_roi_ && img_roi.contains(last_mouse_pos_)) {
            const auto freq_at_pix = frequency_map(last_mouse_pos_);

            if (freq_at_pix != FrequencyMap::value_type(0)) {
                std::ostringstream oss;
                oss << std::round(freq_at_pix) << " Hz";

                cv::putText(display_frame_, oss.str(), last_mouse_pos_, FONT_FACE, FONT_SCALE, cv::Scalar::all(255),
                            THICKNESS);
            }
        }
    }

    /// @brief Prints the ROIs and their associated dominant frequency
    /// @param frequency_map The input frequency map
    void print_rois(const FrequencyMap &frequency_map) {
        frequency_precision_type dominant_freq;
        std::ostringstream oss;
        for (const auto &roi : rois_) {
            cv::rectangle(display_frame_, roi.tl(), roi.br(), cv::Scalar(0, 255, 255));

            oss.str("");
            if (dominant_freq_algo_->compute_dominant_value(frequency_map(roi), dominant_freq))
                oss << dominant_freq << " Hz";
            else
                oss << "N/A";

            cv::putText(display_frame_, oss.str(), roi.br() + cv::Point(MARGIN, 0), FONT_FACE, FONT_SCALE,
                        cv::Scalar::all(255), THICKNESS);
        }
    }

    /// @brief Prints a help message indicating which interactions are possible
    void print_help_message() {
        cv::Point text_pos = help_msg_text_pos_;
        if (display_help_) {
            for (const auto &s : help_msg_) {
                cv::putText(display_frame_, s, text_pos, FONT_FACE, FONT_SCALE, cv::Scalar::all(255), THICKNESS);

                text_pos += cv::Point(0, text_height_ + MARGIN);
            }
        } else {
            static const std::string help_msg = "Press 'h' for help";

            cv::putText(display_frame_, help_msg, help_msg_text_pos_, FONT_FACE, FONT_SCALE, cv::Scalar::all(255),
                        THICKNESS);
        }
    }

    static const std::string WINDOW_NAME;                        ///< Window's name
    static constexpr int FONT_FACE     = cv::FONT_HERSHEY_PLAIN; ///< Font used for text rendering
    static constexpr double FONT_SCALE = 1.0;                    ///< Font scale used for text rendering
    static constexpr int THICKNESS     = 1;                      ///< Line thickness used for text rendering
    static constexpr int MARGIN        = 5;                      ///< Additional space used for text rendering

    int width_;
    int height_;

    std::unique_ptr<Metavision::MTWindow> window_;

    bool is_updating_roi_ = false;
    bool display_help_    = false;

    std::vector<std::string> help_msg_;

    cv::Point dominant_frequency_text_pos_; ///< Position of the dominant frequency text in the image
    cv::Point help_msg_text_pos_;           ///< Position of the help message in the image
    int text_height_;                       ///< Maximum text height in the image

    cv::Mat display_frame_; ///< RGB frame to display

    std::vector<cv::Rect> rois_;  ///< All the ROIs created by the user
    cv::Point crt_roi_start_pos_; ///< Starting position of the ROI being updated

    cv::Point last_mouse_pos_ = {-1, -1}; ///< Last mouse cursor's position in the window

    /// Algorithm used to convert a frequency map to an RGB image
    std::unique_ptr<Metavision::HeatMapFrameGeneratorAlgorithm> freq_map_display_algo_;
    /// Algorithm used to compute the dominant frequency in a frequency map
    std::unique_ptr<Metavision::DominantValueMapAlgorithm<frequency_precision_type>> dominant_freq_algo_;
};

const std::string VibrationGUIStage::WINDOW_NAME = "Vibration estimation";

// Structure for application's parameters
struct Config {
    // Camera's parameters
    std::string raw_file_path_;

    // Vibration estimation algorithm's parameters
    Metavision::FrequencyEstimationConfig vibration_config_;

    // Vibration display's parameters
    float freq_precision_;
    float update_freq_;
    int min_pixel_count_;
    bool no_display;
    bool realtime_playback_speed;
};

bool parse_command_line(int argc, char *argv[], Config &config) {
    const std::string program_desc(
        "Code sample displaying the frequency map of all the vibrating objects seen by the camera. If there is "
        "nothing displayed, please make sure a vibrating object has well been placed in front of the camera.\n");

    po::options_description options_desc;
    po::options_description base_options("Base options");
    // clang-format off
    base_options.add_options()
        ("help,h", "Produce help message.")
        ("input-raw-file,i", po::value<std::string>(&config.raw_file_path_), "Path to input RAW file. If not specified, the camera live stream is used.")
        ("no-display,d", po::bool_switch(&config.no_display)->default_value(false), "Disable output display window")
        ("realtime-playback-speed", po::value<bool>(&config.realtime_playback_speed)->default_value(true), "Replay events at speed of recording if true, otherwise as fast as possible")
        ;
    // clang-format on

    po::options_description estimation_options("Estimation mode options");
    // clang-format off
    estimation_options.add_options()
        ("min-freq",        po::value<float>(&config.vibration_config_.min_freq_)->default_value(10.), "Minimum detected frequency")
        ("max-freq",        po::value<float>(&config.vibration_config_.max_freq_)->default_value(150.), "Maximum detected frequency")
        ("filter-length",   po::value<std::uint32_t>(&config.vibration_config_.filter_length_)->default_value(7), "Number of successive periods to detect a vibration")
        ("max-period-diff", po::value<std::uint32_t>(&config.vibration_config_.diff_thresh_us_)->default_value(1500), "Period stability threshold - the maximum difference (in us) between two periods to be considered the same")
        ("update-freq",     po::value<float>(&config.update_freq_)->default_value(30.f), "Algorithm's update frequency (in Hz)")
        ("freq-precision",  po::value<float>(&config.freq_precision_)->default_value(1.0), "Precision of frequency computation - Width of frequency bins in histogram (in Hz)")
        ("min-pixel-count", po::value<int>(&config.min_pixel_count_)->default_value(25), "Minimum number of pixels to consider a frequency \"real\", i.e. not coming from noise")
        ;
    // clang-format on

    options_desc.add(base_options).add(estimation_options);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return false;
    }

    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return false;
    }

    return true;
}

int main(int argc, char *argv[]) {
    Config conf_;

    if (!parse_command_line(argc, argv, conf_))
        return 1;

    // Keep the start time of execution
    const auto start = std::chrono::high_resolution_clock::now();

    Metavision::Pipeline p;

    // Initialize the camera
    Metavision::Camera camera;
    if (conf_.raw_file_path_.empty()) {
        try {
            camera = Metavision::Camera::from_first_available();
        } catch (const Metavision::CameraException &e) {
            MV_LOG_ERROR() << e.what();
            return 1;
        }
    } else {
        camera = Metavision::Camera::from_file(conf_.raw_file_path_, conf_.realtime_playback_speed);
    }

    const unsigned short width  = camera.geometry().width();
    const unsigned short height = camera.geometry().height();

    // Pipeline
    //
    //  0 (Camera) -->-- 1 (Vibration Estimation) -->-- 2 (Vibration GUI)
    //

    // 0) Camera stage
    auto &cam_stage = p.add_stage(std::make_unique<Metavision::CameraStage>(std::move(camera)));
    cam_stage.detach();

    // 1) Vibration estimation algorithm stage
    auto &vibration_algo_stage = p.add_stage(
        std::make_unique<FrequencyMapAsyncAlgorithmStage>(width, height, conf_.vibration_config_, conf_.update_freq_),
        cam_stage);
    vibration_algo_stage.detach();

    if (conf_.no_display) {
        p.run();
    } else {
        // 2) Vibration GUI
        auto &vibration_gui_stage =
            p.add_stage(std::make_unique<VibrationGUIStage>(width, height, conf_.vibration_config_.min_freq_,
                                                            conf_.vibration_config_.max_freq_, conf_.freq_precision_,
                                                            conf_.min_pixel_count_),
                        vibration_algo_stage);

        // We are not using the display stage here but the VibrationGuiStage instead.
        // We need to call Metavision::EventLoop::poll_and_dispatch() regularly to dispatch system input events to the
        // Vibration GUI. Let's run the pipeline step by step:
        while (p.step())
            Metavision::EventLoop::poll_and_dispatch();
    }

    // Estimate the total time of execution
    const auto end     = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MV_LOG_INFO() << "Ran in" << static_cast<float>(elapsed.count()) / 1000.f << "s";

    return 0;
}
