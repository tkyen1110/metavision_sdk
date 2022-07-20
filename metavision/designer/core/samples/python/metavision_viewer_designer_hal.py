#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This script displays the CD stream from an event-based device
"""

from os import path

from metavision_hal import DeviceDiscovery
from metavision_designer_engine import Controller, KeyboardEvent
from metavision_designer_core import HalDeviceInterface, CdProducer, FrameGenerator, ImageDisplayCV


def parse_args():
    import argparse
    """Defines and parses input arguments"""

    description = "Simple viewer to stream events from an event-based device or RAW file, using " + \
        "Metavision Designer Python API."

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input-raw-file', dest='input_filename', metavar='INPUT_FILENAME',
                        help='Path to input RAW file. If not specified, the camera live stream is used.')

    live_camera_args = parser.add_argument_group('Live camera input parameters '
                                                 '(not compatible with \'--filename\' flag)')
    live_camera_args.add_argument('-s', '--serial', dest='serial', metavar='ID', default='',
                                  help='Serial ID of the camera. If not provided, the first available device will '
                                  'be opened.')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    from_file = False
    if args.input_filename:

        # Check input arguments compatibility
        if args.serial:
            print("Error: flag --serial and --filename are not compatible.")
            return 1

        # Check provided input file exists
        if not(path.exists(args.input_filename) and path.isfile(args.input_filename)):
            print("Error: provided input path '{}' does not exist or is not a file.".format(args.input_filename))
            return 1

        # Open file
        device = DeviceDiscovery.open_raw_file(args.input_filename)
        if not device:
            print("Error: could not open file '{}'.".format(args.input_filename))
            return 1

        from_file = True
    else:
        # Open camera
        device = DeviceDiscovery.open(args.serial)
        if not device:
            print("Could not open camera. Make sure you have an event-based device plugged in")
            return 1

    # Create the controller
    controller = Controller()  # equivalent to controller = Controller(parallel_threads = True),
    # or controller = Controller(True)

    # Create HalDeviceInterface, poll camera buffer every millisecond
    hal_device_interface = HalDeviceInterface(device)
    controller.add_device_interface(hal_device_interface)

    # Create the filtering chains
    # Read CD events from the camera
    prod_cd = CdProducer(hal_device_interface)
    controller.add_component(prod_cd)

    # Generate a graphical representation of the events
    frame_generator = FrameGenerator(prod_cd)  # Or you can also specify the intensity for on/off events
    # (on a gray background), like this :
    # frame_generator = FrameGenerator(prod_cd, on_intensity = 1., off_intensity = 0.)
    frame_generator.set_name("CD FrameGenerator")
    controller.add_component(frame_generator)

    # Display the generated image
    img_display = ImageDisplayCV(frame_generator)
    img_display.set_name("CD Events Display")
    controller.add_component(img_display)

    # Setup rendering with 25 frames per second
    controller.add_renderer(img_display, Controller.RenderingMode.SlowestClock, 25.)
    controller.enable_rendering(True)

    # Set controller parameters for running :
    controller.set_slice_duration(10000)
    controller.set_batch_duration(100000)
    controller.set_sync_mode(Controller.SyncMode.CameraTiming)
    do_sync = True if from_file else False

    # Start the streaming of events
    i_events_stream = device.get_i_events_stream()
    i_events_stream.start()

    # Start the camera
    if not from_file:
        simple_device = device.get_i_device_control()
        simple_device.start()

    # Main program loop
    while not (controller.is_done()):

        # Run the simulation
        controller.run(do_sync)

        if (not args.input_filename) and (hal_device_interface.get_latest_error() < 0):
            print("Error: camera unplugged")
            return 1

        # Get the last key pressed
        last_key = controller.get_last_key_pressed()

        # Exit program if requested
        if last_key == ord('q') or last_key == KeyboardEvent.Symbol.Escape:
            break

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
