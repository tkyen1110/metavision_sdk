#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to track general objects.
"""

from metavision_hal import DeviceDiscovery
import metavision_designer_engine as mvd_engine
import metavision_designer_core as mvd_core
import metavision_designer_analytics as mvd_analytics


def parse_args():
    import argparse
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Tracking sample.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-raw-file', dest='input_path',
                        help='Path to input RAW file. If not specified, the camera live stream is used.')
    parser.add_argument('--min-size', dest='min_size', type=int,
                        default=10, help='Minimal size of an object to track (in pixels)')
    parser.add_argument('--max-size', dest='max_size', type=int,
                        default=300, help='Maximal size of an object to track (in pixels)')
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Open a live camera if no file is provided
    if args.input_path is None:
        device = DeviceDiscovery.open("")
        if device is None:
            print('Error: Failed to open a camera.')
            return 1
    else:
        device = DeviceDiscovery.open_raw_file(
            args.input_path)

    interface = mvd_core.HalDeviceInterface(device, 1e-3)
    prod_cd = mvd_core.CdProducer(interface)

    # Get the sensor size.
    geometry = device.get_i_geometry()
    width = geometry.get_width()
    height = geometry.get_height()

    # Create a tracking filter using the default configuration
    tracker_config = mvd_analytics.TrackingConfig()
    tracker = mvd_analytics.TrackingFilter(prod_cd, width, height, tracker_config)
    tracker.set_name("Tracker")
    tracker.set_min_size(args.min_size)
    tracker.set_max_size(args.max_size)

    # Generate a graphical representation of the events and the tracking results
    frame_generator = mvd_analytics.TrackingFrameGenerator(
        prod_cd, tracker, width, height)
    frame_generator.set_name("Tracking frame generator")

    # Display the generated image.
    display = mvd_core.ImageDisplayCV(frame_generator, False)
    display.set_name("Tracking display")

    # Create the controller.
    controller = mvd_engine.Controller()
    sync_controller = True

    # Start the streaming of events
    i_events_stream = device.get_i_events_stream()
    i_events_stream.start()

    # Start the camera
    if args.input_path is None:
        simple_device = device.get_i_device_control()
        simple_device.start()
        sync_controller = False

    # Register the components
    controller.add_device_interface(interface)
    controller.add_component(prod_cd)

    controller.add_component(tracker)
    controller.add_component(frame_generator)
    controller.add_component(display)

    # Setup rendering at 25 frames per second
    controller.add_renderer(display, mvd_engine.Controller.RenderingMode.SimulationClock, 25.)
    controller.enable_rendering(True)

    controller.set_slice_duration(10000)
    controller.set_batch_duration(100000)

    # Main loop
    done = False
    cnt = 0

    while not (done or controller.are_producers_done()):
        controller.run(sync_controller)

        last_key = controller.get_last_key_pressed()
        if last_key == ord('q') or last_key == mvd_engine.KeyboardEvent.Symbol.Escape:
            break

        if cnt % 100 == 0:
            controller.print_stats(True)

        cnt = cnt + 1


if __name__ == "__main__":
    main()
