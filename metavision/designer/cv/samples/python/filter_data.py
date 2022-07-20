#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This script displays the CD stream from an event-based device
"""

from metavision_hal import DeviceDiscovery
from metavision_designer_engine import Controller, KeyboardEvent
from metavision_designer_core import HalDeviceInterface, CdProducer, FrameGenerator, RoiFilter, FlipXFilter, FlipYFilter, ImageDisplayCV
from metavision_designer_cv import TrailFilter


def parse_args():
    import argparse
    """Defines and parses input arguments"""

    description = "Simple sample to show how to add filters to your pipeline using the Metavision Designer Python API."

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--roi', default=False, action='store_true',
                        help='Add ROI in the middle of the image')
    parser.add_argument('--trail-filter', default=False, action='store_true',
                        help='Add TrailFilter in the pipeline')
    parser.add_argument('--flip-x', default=False, action='store_true',
                        help='Add FlipXFilter in the pipeline')
    parser.add_argument('--flip-y', default=False, action='store_true',
                        help='Add FlipYFilter in the pipeline')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Open camera
    device = DeviceDiscovery.open("")
    if not device:
        print("Could not open camera. Make sure you have an event-based device plugged in")
        return 1

    # Get the geometry
    i_geometry = device.get_i_geometry()
    width = i_geometry.get_width()
    height = i_geometry.get_height()

    # Create the controller
    controller = Controller()

    # Create HalDeviceInterface, poll camera buffer every millisecond
    hal_device_interface = HalDeviceInterface(device)
    controller.add_device_interface(hal_device_interface)

    # Create the filtering chains
    # Read CD events from the camera
    prod_cd = CdProducer(hal_device_interface)
    controller.add_component(prod_cd)

    next_input = prod_cd

    if args.roi:
        roi_width = int(width / 2)
        roi_height = int(height / 2)
        x0 = int(roi_width / 2)
        y0 = int(roi_height / 2)
        roi_filter = RoiFilter(next_input, x0, y0, x0 + roi_width, y0 + roi_height)
        controller.add_component(roi_filter)
        next_input = roi_filter

    if args.trail_filter:
        trail_filter = TrailFilter(next_input, 50000)
        controller.add_component(trail_filter)
        next_input = trail_filter

    if args.flip_x:
        flipx_filter = FlipXFilter(next_input)
        controller.add_component(flipx_filter)
        next_input = flipx_filter

    if args.flip_y:
        flipy_filter = FlipYFilter(next_input)
        controller.add_component(flipy_filter)
        next_input = flipy_filter

    # Generate a graphical representation of the events
    frame_generator = FrameGenerator(next_input)
    frame_generator.set_name("CD FrameGenerator")
    controller.add_component(frame_generator)

    # Display the generated image
    img_display = ImageDisplayCV(frame_generator)
    img_display.set_name("CD Events Display")
    controller.add_component(img_display)

    # Setup rendering with 25 frames per second
    controller.add_renderer(img_display, Controller.RenderingMode.SlowestClock, 25.)
    controller.enable_rendering(True)

    # Start the camera
    i_events_stream = device.get_i_events_stream()
    simple_device = device.get_i_device_control()
    i_events_stream.start()
    simple_device.start()

    # Set controller parameters for running :
    controller.set_slice_duration(10000)
    controller.set_batch_duration(100000)

    # Main program loop
    while not (controller.is_done()):

        # Get the last key pressed
        last_key = controller.get_last_key_pressed()

        # Exit program if requested
        if last_key == ord('q') or last_key == KeyboardEvent.Symbol.Escape:
            break

        # Run the simulation
        controller.run(False)


if __name__ == '__main__':
    import sys
    sys.exit(main())
