#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Simple script to track simple, non colliding objects.
"""

from metavision_hal import DeviceDiscovery
import metavision_designer_engine as mvd_engine
import metavision_designer_core as mvd_core
import metavision_designer_analytics as mvd_analytics


def parse_args():
    import argparse
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Spatter tracker sample.')
    parser.add_argument('-i', '--input-raw-file', dest='input_path',
                        help='Path to input RAW file. If not specified, the camera live stream is used.')
    parser.add_argument('--cell-width',
                        help='Cell width used for clustering (in pixels)', type=int, default=7)
    parser.add_argument('--cell-height',
                        help='Cell height used for clustering (in pixels)', type=int, default=7)
    parser.add_argument('--processing-accumulation-time',
                        help='Processing accumulation time (in us)', type=int, default=5000)
    parser.add_argument('--out-video',
                        help='Path and name for saving the output slow motion AVI video', type=str)
    parser.add_argument('--video-fps',
                        help='Change the number of frame per second for the video', type=int, default=25.)
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

    # Create a spatter tracker filter
    spatter_tracker = mvd_analytics.SpatterTrackerFilter(prod_cd,
                                                         args.cell_width, args.cell_height,
                                                         args.processing_accumulation_time)
    spatter_tracker.set_name("Spatter Tracker")

    # Generate a graphical representation of the events with the clusters
    frame_generator = mvd_analytics.SpatterFrameGenerator(prod_cd, spatter_tracker)
    frame_generator.set_name("Spatter frame generator")

    # Display the generated image.
    display = mvd_core.ImageDisplayCV(frame_generator, False)
    display.set_name("Spatter Tracker display")

    out_video = args.out_video if args.out_video is not None else ""
    video_writer = mvd_core.VideoWriter(frame_generator, out_video, args.video_fps)
    video_writer.enable(args.out_video is not None)

    # Create the controller.
    controller = mvd_engine.Controller()

    # Register the components
    controller.add_device_interface(interface)
    controller.add_component(prod_cd)
    controller.add_component(spatter_tracker)
    controller.add_component(frame_generator)
    controller.add_component(display)
    controller.add_component(video_writer)
    controller.add_renderer(video_writer, mvd_engine.Controller.RenderingMode.SimulationClock, args.video_fps)

    # Setup rendering at 25 frames per second
    controller.add_renderer(display, mvd_engine.Controller.RenderingMode.SimulationClock, 25.)
    controller.enable_rendering(True)

    # Set controller parameters for running :
    controller.set_slice_duration(args.processing_accumulation_time)
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

    print("Number of tracked clusters: {}".format(spatter_tracker.get_cluster_count()))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
