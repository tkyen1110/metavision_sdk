#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This script displays the CD stream from an event-based DAT file
"""

from os import path

from metavision_designer_engine import Controller, KeyboardEvent
from metavision_designer_core import FileProducer, FrameGenerator, ImageDisplayCV


def parse_args():
    import argparse
    """Defines and parses input arguments"""

    description = "Simple viewer to stream events from a DAT file, using the Metavision Designer Python API."

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input-file', dest='input_filename', metavar='INPUT_FILENAME', required=True,
                        help='Path to a DAT file to read.')
    parser.add_argument('--start-time', dest='start_time', metavar='TIME', type=int,
                        help='Time at which to start the replay (in s)')
    parser.add_argument('--loop', default=False, action='store_true',
                        help='Plays the file in a loop.')

    return parser.parse_args()


def main():
    """Main function"""

    args = parse_args()

    # Check provided input file exists
    if not(path.exists(args.input_filename) and path.isfile(args.input_filename)):
        print("Error: provided input path '{}' does not exist or is not a file.".format(args.input_filename))
        return 1

    prod_cd = FileProducer(args.input_filename, loop=args.loop)

    # If requested, start the replay at args.start_time seconds
    if args.start_time:
        prod_cd.start_at_time(1000000 * args.start_time)  # need to convert to us

    # Generate a graphical representation of the events
    frame_generator = FrameGenerator(prod_cd)
    frame_generator.set_dt(10000)  # Set accumulation time

    # Display the generated image
    img_display = ImageDisplayCV(frame_generator)
    img_display.set_name("CD Events Display")

    # Create the controller
    controller = Controller()

    # Register the filters
    controller.add_component(prod_cd, name="CD Producer")
    controller.add_component(frame_generator, name="CD FrameGenerator")
    controller.add_component(img_display)

    # Setup rendering with 25 frames per second
    controller.add_renderer(img_display, Controller.RenderingMode.SlowestClock, 25.)
    controller.enable_rendering(True)

    # Set controller parameters
    controller.set_slice_duration(10000)
    controller.set_batch_duration(100000)
    controller.set_sync_mode(Controller.SyncMode.CameraTiming)

    # Main program loop
    while not (controller.is_done()):

        # Get the last key pressed
        last_key = controller.get_last_key_pressed()

        # Exit program if requested
        if last_key == ord('q') or last_key == KeyboardEvent.Symbol.Escape:
            break

        controller.run()

    # print stats about time usage
    controller.print_stats()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
