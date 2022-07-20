#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to estimate the frequency of vibrating objects.
It will display a window with a visualization of the events, and
another with the frequency for pixels with periodic motion.
You can use it for example, with the file monitoring_40_50hz.raw
from Metavision Datasets that can be downloaded from our documentation.
"""

from metavision_hal import DeviceDiscovery
import metavision_designer_engine as mvd_engine
import metavision_designer_core as mvd_core
import metavision_designer_analytics as mvd_analytics


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Vibration estimation sample.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-raw-file', dest='input_path',
                        help='Path to input RAW file. If not specified, the camera live stream is used.')
    parser.add_argument(
        '--min-freq-hz',
        dest='min_freq_hz',
        type=float,
        default=10,
        help='Minimum frequency detected.')
    parser.add_argument(
        '--max-freq-hz',
        dest='max_freq_hz',
        type=float,
        default=150,
        help='Minimum frequency detected.')
    parser.add_argument('--filter-length', dest='filter_length', type=int, default=7,
                        help='Number of successive periods to detect a vibration.')
    parser.add_argument('--diff-threshold', dest='diff_threshold', type=int, default=1500,
                        help='Maximum difference between two periods to be considered the same (in us).')
    parser.add_argument('--update-freq', dest='update_freq_hz', default=25,
                        type=float, help='Update frequency of the algorithm.')
    parser.add_argument('--freq-precision', dest='freq_precision_hz', type=float, default=1,
                        help='Precision of frequency calculation - Width of frequency bins in histogram (in Hz).')
    parser.add_argument('--min-pixel-count', dest='min_pixel_count', type=int, default=25,
                        help='Minimum number of pixels to consider a frequency "real", i.e not coming from noise')
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Open a live camera if no file is provided
    from_file = False
    if args.input_path is None:
        device = DeviceDiscovery.open("")
        if device is None:
            print('Error: Failed to open a camera.')
            return 1
    else:
        device = DeviceDiscovery.open_raw_file(args.input_path)
        from_file = True

    # Create HalDeviceInterface
    interface = mvd_core.HalDeviceInterface(device)

    # Read CD events from the camera
    prod_cd = mvd_core.CdProducer(interface)

    # Estimate the frequency map
    frequency_estimator = mvd_analytics.FrequencyMapFilter(
        prod_cd,
        args.update_freq_hz,
        args.filter_length,
        args.min_freq_hz,
        args.max_freq_hz,
        args.diff_threshold)
    frequency_estimator.set_name("Frequency Map Estimator")

    # Generate the frequency map to be displayed.
    vibration_displayer = mvd_analytics.FrequencyMapFrameGenerator(
        frequency_estimator,
        args.update_freq_hz,
        args.min_freq_hz,
        args.max_freq_hz,
        args.freq_precision_hz,
        args.min_pixel_count)
    vibration_displayer.set_name("Frequency map frame generator")

    # Display the generated image.
    img_display_vib = mvd_core.ImageDisplayCV(vibration_displayer)
    img_display_vib.set_name("Vibration frequency")

    # Generate a graphical representation of the events, for reference.
    frame_generator_cd = mvd_core.FrameGenerator(prod_cd)
    frame_generator_cd.set_name("CD FrameGenerator")

    # Display the input events image.
    img_display_cd = mvd_core.ImageDisplayCV(frame_generator_cd)
    img_display_cd.set_name("Input CD events")

    # Create the controller.
    controller = mvd_engine.Controller()

    # Register the components
    controller.add_device_interface(interface)
    controller.add_component(prod_cd)
    controller.add_component(frame_generator_cd, "cd_frame_generator")
    controller.add_component(img_display_cd, "img_display_cd")
    controller.add_component(frequency_estimator, "frequency_estimator")
    controller.add_component(vibration_displayer, "freq_map_frame_generator")
    controller.add_component(img_display_vib, "img_display_vib")

    # Set up rendering at 25 frames per second
    controller.add_renderer(img_display_cd, mvd_engine.Controller.RenderingMode.SimulationClock, 25.)
    controller.add_renderer(img_display_vib, mvd_engine.Controller.RenderingMode.SimulationClock, 25.)
    controller.enable_rendering(True)

    # Set controller parameters for running :
    controller.set_slice_duration(10000)
    controller.set_batch_duration(100000)
    sync_controller = True if from_file else False

    # Start the streaming of events
    i_events_stream = device.get_i_events_stream()
    i_events_stream.start()

    # Start the camera
    if not from_file:
        simple_device = device.get_i_device_control()
        simple_device.start()

    # Main loop
    cnt = 0
    while not (controller.is_done()):
        controller.run(sync_controller)
        last_key = controller.get_last_key_pressed()
        if last_key == ord('q') or last_key == mvd_engine.KeyboardEvent.Symbol.Escape:
            break
        if cnt % 100 == 0:
            controller.print_stats()
        cnt = cnt + 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
