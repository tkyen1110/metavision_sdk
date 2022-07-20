/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_CONFIG_H
#define METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_CONFIG_H

#include <istream>
#include <limits>

#include "metavision/sdk/base/utils/timestamp.h"

namespace Metavision {

/// @brief Structure used to configure the @ref TrackingAlgorithm.
struct TrackingConfig {
    bool print_timings_ = false; ///< If enabled, displays a profiling summary.

    /// @brief Defines the type of cluster maker used by the @ref TrackingAlgorithm.
    enum class ClusterMaker { SIMPLE_GRID, MEDOID_SHIFT };

    ClusterMaker cluster_maker_ = ClusterMaker::SIMPLE_GRID; ///< Type of cluster maker to use.

    // Grid cluster related parameters.
    int cell_width_           = 10;   ///< Grid clustering cell's width (in pixels).
    int cell_height_          = 10;   ///< Grid clustering cell's height (in pixels).
    timestamp cell_deltat_    = 1000; ///< Grid clustering cell's time delta (in us).
    int activation_threshold_ = 5;    ///< Minimum number of events needed to activate the grid.

    // Medoid shift cluster related parameters.
    /// Maximum spatial distance (using Manhattan distance) for two events to be in the same cluster.
    float medoid_shift_spatial_dist_ = 5;
    /// Maximum temporal distance for two events to be in the same cluster.
    timestamp medoid_shift_temporal_dist_ = 10000;
    /// Minimum width and height for a cluster to be considered valid and given to the tracking engine.
    int medoid_shift_min_size_ = 2;

    /// @brief Defines type of data association used by the @ref TrackingAlgorithm
    enum class DataAssociation { NEAREST, IOU };

    DataAssociation data_association_ = DataAssociation::IOU; ///< Type of data association to use.

    // Nearest data association related parameters.
    /// Maximum distance between cluster's centroid and tracker's position.
    double max_dist_ = 150.;

    // IOU data association related parameters.
    /// Maximum distance between cluster's centroid and tracker's position.
    double iou_max_dist_ = 150.;

    /// @brief Defines the type of initializer used by the @ref TrackingAlgorithm. Only one method implemented for now
    enum class Initializer { SIMPLE };

    // Simple initializer related parameters.
    Initializer initializer_         = Initializer::SIMPLE; ///< Type of initializer to use.
    int simple_initializer_min_size_ = 0;
    int simple_initializer_max_size_ = std::numeric_limits<std::uint16_t>::max();

    /// @brief Defines the type of motion model used by the @ref TrackingAlgorithm.
    enum class MotionModel { SIMPLE, INSTANT, SMOOTH, KALMAN };

    MotionModel motion_model_ = MotionModel::SIMPLE; ///< Type of motion model to use.

    // Smooth motion model related parameters.
    double smooth_mm_alpha_vel_ = 0.001;
    bool smooth_mm_is_postponed = false;

    // Kalman motion model related parameters.
    /// @brief Defines the Kalman motion model used by the @ref TrackingAlgorithm.
    enum class KalmanModel { CONSTANT_VELOCITY, CONSTANT_ACCELERATION };

    /// @brief Defines the Kalman policy model used by the @ref TrackingAlgorithm.
    enum class KalmanPolicy { ADAPTIVE_NOISE, MEASUREMENT_TRUST };

    /// Standard deviation of the transition noise for the tracker's position.
    double kalman_motion_model_pos_trans_std_ = 0.0001;
    /// Standard deviation of the transition noise for the tracker's velocity.
    double kalman_motion_model_vel_trans_std_ = 0.05;
    /// Standard deviation of the transition noise for the tracker's acceleration.
    double kalman_motion_model_acc_trans_std_ = 1e-9;
    /// Standard deviation of the observation noise for the tracker's position.
    double kalman_motion_model_pos_obs_std_ = 100;
    /// Factor to multiply to noise variance at initialization.
    double kalman_motion_model_init_factor_ = 1e12;
    /// Expected average events per pixel rate in events/us.
    double kalman_motion_model_avg_events_per_pixel_ = 1;
    /// Minimal timestep at which to compute Kalman filter.
    timestamp kalman_motion_model_min_dt_ = 1000;
    /// Policy used in the Kalman Filter.
    KalmanPolicy kalman_motion_model_policy_ = KalmanPolicy::ADAPTIVE_NOISE;
    /// Motion model used in the Kalman filter.
    KalmanModel kalman_motion_model_motion_model_ = KalmanModel::CONSTANT_VELOCITY;

    /// @brief Defines the type of tracker used by the @ref TrackingAlgorithm.
    enum class Tracker { ELLIPSE, CLUSTERKF };

    Tracker tracker_ = Tracker::CLUSTERKF; ///< Type of tracker to use.

    // Ellipse tracker related parameters.
    enum class EllipseUpdateFunction { UNIFORM, GAUSSIAN, SIGNED_GAUSSIAN, TRUNCATED_GAUSSIAN };
    enum class EllipseUpdateMethod {
        PER_EVENT,
        ELLIPSE_FITTING,
        GAUSSIAN_FITTING,
        ELLIPSE_FITTING_FULL,
        GAUSSIAN_FITTING_FULL
    };

    double sigma_xx_                       = 5.;
    double sigma_yy_                       = 5.;
    double alpha_pos_                      = 0.1;
    double alpha_shape_                    = 0.04;
    EllipseUpdateFunction update_function_ = EllipseUpdateFunction::GAUSSIAN;
    double update_function_param_          = 100.;
    EllipseUpdateMethod update_method_     = EllipseUpdateMethod::GAUSSIAN_FITTING;
    bool decompose_covariance_             = false;

    // Cluster KF related parameters.
    double cluster_kf_pos_var_      = 1200000000;
    double cluster_kf_vel_var_      = 32000;
    double cluster_kf_acc_var_      = 0.8;
    double cluster_kf_size_var_     = 200000;
    double cluster_kf_vel_size_var_ = 2;
    double cluster_kf_pos_obs_var_  = 200;
    double cluster_kf_size_obs_var_ = 1e3;

    // Tracker eraser related parameters.
    /// Delta ts (in us) for the position prediction.
    timestamp delta_ts_for_prediction_ = 5000;
    /// Time (in us) to wait without having updated a tracker before deleting it.
    timestamp ts_to_stop_            = 100000;
    std::string forbidden_area_file_ = "";

    // Not configurable from configfile (for now)
    double min_speed_ = 0;
    double max_speed_ = std::numeric_limits<float>::max();
};

inline std::istream &operator>>(std::istream &in, TrackingConfig::ClusterMaker &cm) {
    std::string token;
    in >> token;
    if (token == "SIMPLE_GRID")
        cm = TrackingConfig::ClusterMaker::SIMPLE_GRID;
    else if (token == "MEDOID_SHIFT")
        cm = TrackingConfig::ClusterMaker::MEDOID_SHIFT;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::DataAssociation &da) {
    std::string token;
    in >> token;
    if (token == "NEAREST")
        da = TrackingConfig::DataAssociation::NEAREST;
    else if (token == "IOU")
        da = TrackingConfig::DataAssociation::IOU;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::MotionModel &mm) {
    std::string token;
    in >> token;
    if (token == "SIMPLE")
        mm = TrackingConfig::MotionModel::SIMPLE;
    else if (token == "INSTANT")
        mm = TrackingConfig::MotionModel::INSTANT;
    else if (token == "SMOOTH")
        mm = TrackingConfig::MotionModel::SMOOTH;
    else if (token == "KALMAN")
        mm = TrackingConfig::MotionModel::KALMAN;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::Tracker &mm) {
    std::string token;
    in >> token;
    if (token == "ELLIPSE")
        mm = TrackingConfig::Tracker::ELLIPSE;
    else if (token == "CLUSTERKF")
        mm = TrackingConfig::Tracker::CLUSTERKF;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::KalmanPolicy &pol) {
    std::string token;
    in >> token;
    if (token == "ADAPTIVE_NOISE")
        pol = TrackingConfig::KalmanPolicy::ADAPTIVE_NOISE;
    else if (token == "MEASUREMENT_TRUST")
        pol = TrackingConfig::KalmanPolicy::MEASUREMENT_TRUST;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::KalmanModel &mm) {
    std::string token;
    in >> token;
    if (token == "CONSTANT_VELOCITY")
        mm = TrackingConfig::KalmanModel::CONSTANT_VELOCITY;
    else if (token == "CONSTANT_ACCELERATION")
        mm = TrackingConfig::KalmanModel::CONSTANT_ACCELERATION;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::EllipseUpdateFunction &uf) {
    std::string token;
    in >> token;
    if (token == "UNIFORM")
        uf = TrackingConfig::EllipseUpdateFunction::UNIFORM;
    else if (token == "GAUSSIAN")
        uf = TrackingConfig::EllipseUpdateFunction::GAUSSIAN;
    else if (token == "SIGNED_GAUSSIAN")
        uf = TrackingConfig::EllipseUpdateFunction::SIGNED_GAUSSIAN;
    else if (token == "TRUNCATED_GAUSSIAN")
        uf = TrackingConfig::EllipseUpdateFunction::TRUNCATED_GAUSSIAN;
    return in;
}

inline std::istream &operator>>(std::istream &in, TrackingConfig::EllipseUpdateMethod &um) {
    std::string token;
    in >> token;
    if (token == "PER_EVENT")
        um = TrackingConfig::EllipseUpdateMethod::PER_EVENT;
    else if (token == "ELLIPSE_FITTING")
        um = TrackingConfig::EllipseUpdateMethod::ELLIPSE_FITTING;
    else if (token == "GAUSSIAN_FITTING")
        um = TrackingConfig::EllipseUpdateMethod::GAUSSIAN_FITTING;
    else if (token == "ELLIPSE_FITTING_FULL")
        um = TrackingConfig::EllipseUpdateMethod::ELLIPSE_FITTING_FULL;
    else if (token == "GAUSSIAN_FITTING_FULL")
        um = TrackingConfig::EllipseUpdateMethod::GAUSSIAN_FITTING_FULL;
    return in;
}

} // namespace Metavision

#endif // METAVISION_SDK_ANALYTICS_TRACKING_ALGORITHM_CONFIG_H
