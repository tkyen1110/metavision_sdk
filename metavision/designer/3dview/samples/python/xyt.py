#!/usr/bin/env python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This script shows how to use the Image3dDisplayXYT class in order to display events in a XYT space
"""

from metavision_hal import DeviceDiscovery
from metavision_designer_engine import Controller, KeyboardEvent
from metavision_designer_core import HalDeviceInterface, CdProducer
from metavision_designer_cv import SpatioTemporalContrast
from metavision_designer_3dview import Image3dDisplayXYT, CopyAndFrame

from os import path
from time import sleep


def parse_args():
    import argparse
    """Defines and parses input arguments"""

    description = "3D viewer to stream events from an event-based device or RAW file, " + \
        "and display them on a 3D view x-y-t, using the Metavision Designer Python API."

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-i', '--input-raw-file', dest='input_filename', metavar='INPUT_FILENAME',
                        help='Path to input RAW file. If not specified, the camera live stream is used.')
    parser.add_argument('--show-axis-labels', default=False, action='store_true',
                        help='Add the xyt labels.')
    parser.add_argument('--show-frames', default=False, action='store_true',
                        help='Show frames next to events in the 3D view')
    parser.add_argument('-b', '--bias-file', dest='bias_file', default='', help='Path to the bias file')

    live_camera_args = parser.add_argument_group(
        "Live camera input parameters (not compatible with \'--input-raw-file\' flag)")
    live_camera_args.add_argument(
        '-s', '--serial', dest='serial', metavar='ID', default='',
        help='Serial ID of the camera. If not provided, the first available device will be opened.')

    return parser.parse_args()


def get_biases_from_file(path: str):
    """
    Helper function to read bias from a file
    """
    biases = {}
    try:
        biases_file = open(path, 'r')
    except IOError:
        print("Cannot open bias file: " + path)
    else:
        for line in biases_file:

            # Skip lines starting with '%': comments
            if line.startswith('%'):
                continue

            # element 0 : value
            # element 1 : name
            split = line.split("%")
            biases[split[1].strip()] = int(split[0])
    return biases


def main():
    """Main function"""
    args = parse_args()

    from_file = False
    if args.input_filename:
        from_file = True

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
    else:
        # Open camera
        device = DeviceDiscovery.open(args.serial)
        if not device:
            print("Could not open camera. Make sure you have an event-based device plugged in")
            return 1

        bias_file = args.bias_file
        if bias_file:
            i_ll_biases = device.get_i_ll_biases()

            if i_ll_biases is not None:
                biases = get_biases_from_file(bias_file)
                for bias_name, bias_value in biases.items():
                    i_ll_biases.set(bias_name, bias_value)
                print('Biases are set from the file: ' + bias_file)

    # Get the geometry
    i_geometry = device.get_i_geometry()
    width = i_geometry.get_width()
    height = i_geometry.get_height()

    # Create the controller
    controller = Controller()

    # Create an interface to read events from a HAL device
    hal_device_interface = HalDeviceInterface(controller, device)

    # Create the filtering chains
    # Read CD events from the camera
    prod_cd = CdProducer(hal_device_interface)

    # Add STC noise filter
    noise_filter = SpatioTemporalContrast(prod_cd, 5000)

    # Add the xyt 3D view to the pipeline
    if args.show_frames:
        frame_and_evs = CopyAndFrame(noise_filter, CopyAndFrame.halfmode.OUT_OF_BOX, 10, 1000)
        width = width * 2
        display_xyt = Image3dDisplayXYT(frame_and_evs, 2)
    else:
        display_xyt = Image3dDisplayXYT(noise_filter, 2)

    controller.add_renderer(display_xyt, Controller.RenderingMode.SlowestClock, 25)
    display_xyt.set_name("XYT")

    display_xyt.set_projection_matrix_as_perspective(fovy=25.0, ar=1.0, near=0.1, far=100000)
    display_xyt.set_view_matrix_as_lookat(eye=[-(width * 1.2), -(height * 2), -(4 * width)],
                                          center=[0, 0, 1.2 * width], up=[0, -1, 0])
    display_xyt.set_camera_manipulator(Image3dDisplayXYT.CameraManipulator.TRACKBALL)

    positive_events_color = [30. / 255., 126. / 255, 199. / 255]
    negative_events_color = [64. / 255., 37. / 255, 52. / 255]
    display_xyt.set_event_display_colors(positive_events_color, negative_events_color)

    background_color = [1., 1, 1, 1.]
    display_xyt.set_scene_display_parameters(
        Image3dDisplayXYT.SceneDisplayProperty.BACKGROUND_COLOR,
        background_color)

    prism_color = [1, 0.6, 0.6, 1]
    display_xyt.set_scene_display_parameters(Image3dDisplayXYT.SceneDisplayProperty.PRISM_COLOR, prism_color)

    point_size = 2.
    display_xyt.set_scene_display_parameters(Image3dDisplayXYT.SceneDisplayProperty.POINT_SIZE, point_size)

    # Add prism
    display_xyt.add_prism_with_ticks(10)  # you can also specify ticks size (by specifying tick_size = ...)
    # Or else :
    # display_xyt.add_prism()

    # Add label on the axis if requested
    if args.show_axis_labels:
        axis_text_font_size = 30
        axis_text_color = [76. / 255., 81. / 255, 87. / 255, 1.]
        display_xyt.add_text("x", [0, height / 2 + axis_text_font_size, 0],
                             color=axis_text_color, size=axis_text_font_size)
        display_xyt.add_text("y", [-(width / 2 + axis_text_font_size), 0, 0],
                             color=axis_text_color, size=axis_text_font_size)
        display_xyt.add_text("t", [-(width / 2 + axis_text_font_size), height / 2 +
                                   axis_text_font_size, 1000], color=axis_text_color, size=axis_text_font_size)

    # Set the bounding box that will be used to compute the view matrix for the standard camera view.
    # By default, the bounding box is automatically computed whenever a standard view is selected.
    # This can be a problem if the displayed geometry varies a lot, or if the scene is empty at the
    # moment you switch to a standard view.
    # You can call this function to switch the way the standard views are computed to use the given
    # bounding box
    display_xyt.set_standard_views_bounding_box(bounding_box_mins=[-width / 2.0, -height / 2.0, 0],
                                                bounding_box_maxs=[width / 2.0, height / 2.0, 10 * width])

    help_message = "Press :\n\n"
    help_message += "  'u' for UP view\n\n"
    help_message += "  'd' for DOWN view\n\n"
    help_message += "  'l' for LEFT view\n\n"
    help_message += "  'r' for RIGHT view\n\n"
    help_message += "  'b' for BACK view\n\n"
    help_message += "  'f' for FRONT view\n\n"
    help_message += "  'o' for ORIGINAL view\n\n"
    help_message += "  'p' to pause\n\n"
    help_message += "  'h' to show/hide this help message\n\n"
    help_message += "  'q' or Escape key to exit\n"

    help_text_id = display_xyt.add_2D_text(
        help_message, [0, 270], color=[76. / 255., 81. / 255, 87. / 255, 1.], size=13)

    controller.enable_rendering(True)

    i_events_stream = device.get_i_events_stream()
    i_events_stream.start()

    # Start the camera
    if not from_file:
        simple_device = device.get_i_device_control()
        simple_device.start()

    # State variables
    paused = False
    help_message_shown = True

    controller.set_slice_duration(10000)
    controller.set_batch_duration(100000)
    controller.set_sync_mode(Controller.SyncMode.CameraTiming)
    do_sync = True if from_file else False

    # Main program loop
    while not controller.is_done():

        # Get the last key pressed
        last_key = controller.get_last_key_pressed()

        # Exit program if requested
        if last_key == ord('q') or last_key == KeyboardEvent.Symbol.Escape:
            break
        elif last_key == ord('u'):
            display_xyt.set_standard_view(Image3dDisplayXYT.StandardView.UP)
        elif last_key == ord('d'):
            display_xyt.set_standard_view(Image3dDisplayXYT.StandardView.DOWN)
        elif last_key == ord('l'):
            display_xyt.set_standard_view(Image3dDisplayXYT.StandardView.LEFT)
        elif last_key == ord('r'):
            display_xyt.set_standard_view(Image3dDisplayXYT.StandardView.RIGHT)
        elif last_key == ord('b'):
            display_xyt.set_standard_view(Image3dDisplayXYT.StandardView.BACK)
        elif last_key == ord('f'):
            display_xyt.set_standard_view(Image3dDisplayXYT.StandardView.FRONT)
        elif last_key == ord('o'):
            display_xyt.set_standard_view(Image3dDisplayXYT.StandardView.ORIGINAL)
        elif last_key == ord('h'):
            help_message_shown = not help_message_shown
            display_xyt.show_item(help_text_id) if help_message_shown else display_xyt.hide_item(help_text_id)

        elif last_key == ord('p'):
            display_xyt.toggle_pause_status()
            paused = not paused

        if from_file and paused:
            sleep(0.05)
            controller.process_ui_events()
        else:
            controller.run(do_sync)


if __name__ == '__main__':
    import sys
    sys.exit(main())
