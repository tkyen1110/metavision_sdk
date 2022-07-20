#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to estimate and display a sparse optical flow.
"""

from metavision_hal import DeviceDiscovery
from metavision_designer_engine import Controller, KeyboardEvent
from metavision_designer_core import HalDeviceInterface, CdProducer, ImageDisplayCV
from metavision_designer_cv import SpatioTemporalContrast, SparseOpticalFlow, FlowFrameGenerator


def parse_args():
    import argparse
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Sparse optical flow sample.')
    parser.add_argument('-i', '--input-raw-file', dest='input_path',
                        help='Path to input RAW file. If not specified, the camera live stream is used.')
    args = parser.parse_args()
    return args


def main():
    """ Main """
    args = parse_args()

    # Open a live camera if no file is provided
    from_file = False
    if not args.input_path:
        device = DeviceDiscovery.open("")
        if device is None:
            print('Error: Failed to open a camera.')
            return 1
    else:
        device = DeviceDiscovery.open_raw_file(args.input_path)
        from_file = True

    # Create HalDeviceInterface
    interface = HalDeviceInterface(device)

    # Read CD events from the interface
    prod_cd = CdProducer(interface)

    # Add event filtering
    stc_filter = SpatioTemporalContrast(prod_cd, 40000)
    stc_filter.set_name("STC filter")

    # Create a sparse optical flow filter
    sparse_flow = SparseOpticalFlow(
        stc_filter, SparseOpticalFlow.Tuning.FAST_OBJECTS)
    sparse_flow.set_name("Sparse Optical Flow")

    # Generate a graphical representation of the events with the flow
    flow_frame_generator = FlowFrameGenerator(
        prod_cd, sparse_flow, True)
    flow_frame_generator.set_name("Flow frame generator")

    # Display the generated image
    flow_display = ImageDisplayCV(flow_frame_generator, False)
    flow_display.set_name("Flow display")

    # Create the controller
    controller = Controller()

    # Register the components
    controller.add_device_interface(interface)
    controller.add_component(prod_cd)
    controller.add_component(stc_filter)
    controller.add_component(sparse_flow)
    controller.add_component(flow_frame_generator)
    controller.add_component(flow_display)

    # Setup rendering at 25 frames per second
    controller.add_renderer(flow_display, Controller.RenderingMode.SimulationClock, 25.)
    controller.enable_rendering(True)

    # Set controller parameters for running
    controller.set_slice_duration(10000)
    controller.set_batch_duration(100000)
    do_sync = True if from_file else False

    # Main loop
    cnt = 0

    # Start the streaming of events
    i_events_stream = device.get_i_events_stream()
    i_events_stream.start()

    # Start the camera
    if not from_file:
        simple_device = device.get_i_device_control()
        simple_device.start()

    while not controller.are_producers_done():
        controller.run(do_sync)

        last_key = controller.get_last_key_pressed()
        if last_key == ord('q') or last_key == KeyboardEvent.Symbol.Escape:
            break

        if cnt % 10 == 0:
            controller.print_stats(False)

        cnt = cnt + 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
