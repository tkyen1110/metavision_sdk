#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to count small objects.
It will display a window with a visualization of the events and the line counters.
You can use it, for example, with the reference file 80_balls.raw.
"""

from metavision_hal import DeviceDiscovery
import metavision_designer_engine as mvd_engine
import metavision_designer_core as mvd_core
import metavision_designer_analytics as mvd_analytics


def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Counting sample.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i',
        '--input-raw-file',
        dest='input_path',
        help='Path to input RAW file. If not specified, the camera live stream is used.')
    parser.add_argument('-n', '--num-lines', dest='num_lines', type=int, default=4,
                        help='Number of lines for counting between min-y and max-y.')
    parser.add_argument('--min-y', dest='min_y_line', type=int, default=150,
                        help='Ordinate at which to place the first line counter.')
    parser.add_argument('--max-y', dest='max_y_line', type=int, default=330,
                        help='Ordinate at which to place the last line counter.')
    parser.add_argument('--object-min-size', dest='object_min_size', type=float, default=6.,
                        help='Approximate minimum size of an object to count (its largest dimension in mm).')
    parser.add_argument('--object-average-speed', dest='object_average_speed', type=float, default=5.,
                        help='Approximate average speed of an object to count in meters per second.')
    parser.add_argument('--distance-object-camera',
                        dest='distance_object_camera', type=float,
                        default=300.,
                        help='Average distance between the flow of objects to count and the camera (distance in mm).')
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
        device = DeviceDiscovery.open_raw_file(args.input_path)
    interface = mvd_core.HalDeviceInterface(device, 1e-3)
    prod_cd = mvd_core.CdProducer(interface)
    # Get the sensor size.
    geometry = device.get_i_geometry()
    width = geometry.get_width()
    height = geometry.get_height()

    # Keep only OFF events
    off_cd = mvd_core.PolarityFilter(prod_cd, 0)
    off_cd.set_name("Polarity Filter ")
    # Count particles using CD events
    counting_algorithm = mvd_analytics.CountingFilter(
        off_cd,
        width,
        height,
        args.object_min_size,
        args.object_average_speed,
        args.distance_object_camera
    )
    counting_algorithm.set_name("Counting Algorithm ")
    # Generate the counting results to be displayed on top of events.
    counting_displayer = mvd_analytics.CountingFrameGenerator(
        prod_cd,
        counting_algorithm,
        width,
        height)
    counting_displayer.set_name("Counting frame generator")
    # Add line counters
    if args.max_y_line < 0 or args.max_y_line >= height:
        print('Error : Invalid argument max-y : {}'.format(args.max_y_line))
        print("        Expect value inside [0, {}[ ".format(height))
        return 1
    if args.min_y_line < 0 or args.min_y_line >= args.max_y_line:
        print('Error : Invalid argument min-y : {}'.format(args.min_y_line))
        print("        Expect value inside [0, {}[ ".format(args.max_y_line))
        return 1
    y_step = int((args.max_y_line - args.min_y_line) / (args.num_lines - 1))
    y = int(args.min_y_line)
    for k in range(args.num_lines):
        counting_algorithm.add_line_counter(y)
        counting_displayer.add_line_counter(y)
        y += y_step

    # Display the generated image.
    img_display_count = mvd_core.ImageDisplayCV(
        counting_displayer, False)
    img_display_count.set_name("Counting")
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
    controller.add_component(off_cd, "off_events")
    controller.add_component(counting_algorithm, "counting_algorithm")
    controller.add_component(counting_displayer, "counting_displayer")
    controller.add_component(img_display_count, "img_display_count")
    # Set up rendering at 25 frames per second
    controller.add_renderer(img_display_count,
                            mvd_engine.Controller.RenderingMode.SimulationClock, 25.)
    controller.enable_rendering(True)

    controller.set_slice_duration(10000)
    controller.set_batch_duration(100000)

    # Main loop
    cnt = 0
    while not (controller.is_done()):
        controller.run(sync_controller)
        last_key = controller.get_last_key_pressed()
        if last_key == ord('q') or last_key == mvd_engine.KeyboardEvent.Symbol.Escape:
            break
        if cnt % 100 == 0:
            controller.print_stats(True)
        cnt = cnt + 1


if __name__ == "__main__":
    main()
