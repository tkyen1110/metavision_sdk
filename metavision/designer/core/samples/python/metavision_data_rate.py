#!/usr/bin/python

# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
This script displays the CD stream from an event-based device and the data rate
"""

import os
from os import path
import argparse
from matplotlib import pyplot as plt
import numpy as np

import metavision_designer_engine as mvd_engine
import metavision_designer_core as mvd_core
import metavision_hal as mv_hal


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
        description='Play events from RAW or DAT file')
    parser.add_argument('-i', '--input-file', dest='input_filename',
                        required=True, default='', help='Input file path')
    parser.add_argument('-s', '--step-duration', type=int, default=10000,
                        help='Controller step size in us. Also defines data rate temporal resolution.\
                            (Default = %(default)s')
    parser.add_argument('-r', '--render-period', type=int, default=80000,
                        help='Controller batch size in us. Also defines the events display render period.\
                            (Default = %(default)s')
    parser.add_argument('-t', '--time-window', type=int, default=10,
                        help='Data rate time window in seconds. (Default=%(default)s)')
    parser.add_argument('-d', '--data-rate-render-period', type=int, default=500000,
                        help='Data rate plot render period in us. (Default = %(default)s)')
    parser.add_argument('--frame-dir-output', type=str, default='',
                        help='Folder where to store the generated frames for video conversion')
    parser.add_argument('-y', '--y-max-evrate', type=float, default=20.,
                        help='Max event rate (MEv/s) for Y axis.\
                        (Default = %(default)s.)')

    args = parser.parse_args()
    return args


class EventFrameDisplay:
    """
    Simple wrapper around Python Consumer
    Manage framed data (analog_buffer) callbacks from designer pipelines
    """

    def __init__(self, producer_name, frame_gen_name, step=10000):
        """
        Constructor: default display to openCV
        Matplotlib display can be activated by caller
        """
        self.__producer_name = producer_name
        self.__frame_gen_name = frame_gen_name
        self.__frame = None
        self.__disp = None
        self.__evrate = []
        self.__time = []
        self.__step = step

    def event_callback(self, ts, src_events, src_2D_arrays):
        """
        Python Callback for PythonConsumer component in a Metavision Designer pipeline
        """

        # frame generation
        if self.__frame_gen_name in src_2D_arrays:
            # get frame data from the analog buffer
            frame_buffer = src_2D_arrays[self.__frame_gen_name][2]

            # matplotlib requires RGB format, our buffer is in BGR, we need to convert
            rbg = frame_buffer[..., ::-1]
            self.__frame = rbg.squeeze()

        # data rate computation
        if self.__producer_name in src_events:
            # get the event list from the event_buffer
            event_buffer = src_events[self.__producer_name][2]
            # the number of events in this slice
            ev_slice_count = len(event_buffer)
            # time normalization to get the rate
            ev_slice_rate = 1.*ev_slice_count/self.__step
            self.__evrate.append(ev_slice_rate)  # we store the rate
            self.__time.append(1e-6*ts)  # and the timestamp (for plotting)

    def pop_rate_series(self):
        """
        Get the rates and reset them
        """
        out = (self.__time, self.__evrate)
        self.__time = []
        self.__evrate = []
        return out

    def get_frame(self):
        """
        Called from main thread to Display frame
        """
        return self.__frame


def main():
    """Play event data and visualize their data rate"""

    # Parse command line arguments
    args = parse_args()

    # Check validity of input arguments
    if not(path.exists(args.input_filename) and path.isfile(args.input_filename)):
        print("Error: provided input path '{}' does not exist or is not a file.".format(
            args.input_filename))
        return 1

    is_raw = args.input_filename.endswith('.raw')
    is_dat = args.input_filename.endswith('.dat')

    if not (is_raw or is_dat):
        print("Error: provided input path '{}' does not have the right extension. ".format(args.input_filename) +
              "It has either to be a .raw or a .dat file")
        return 1

    # Create the pipeline
    controller = mvd_engine.Controller()

    if is_dat:
        cd_producer = mvd_core.FileProducer(args.input_filename)
    else:  # RAW file
        device = mv_hal.DeviceDiscovery.open_raw_file(args.input_filename)
        if not device:
            print("Error: could not open file '{}'.".format(args.input_filename))
            return 1

        # Add the device interface to the pipeline
        interface = mvd_core.HalDeviceInterface(device)
        controller.add_device_interface(interface)

        cd_producer = mvd_core.CdProducer(interface)

        # Start the streaming of events
        i_events_stream = device.get_i_events_stream()
        i_events_stream.start()

    # Add CD producer to the pipeline
    controller.add_component(cd_producer, "CD Producer")

    # Create Frame Generator with 20ms accumulation time
    frame_gen = mvd_core.FrameGenerator(cd_producer)
    frame_gen.set_dt(20000)
    controller.add_component(frame_gen, "FrameGenerator")

    # We use PythonConsumer to get the frame generated by the FrameGenerator
    # and to compute the data rate by getting the event buffer from the CDProducer.
    # Then pyconsumer will callback the application each time it receives data, using the event_callback function.
    # The names of the two producers is used here to distinguish which events we are processing
    prod_cd_string = "ProdCD"
    frame_gen_string = "FrameGen"
    ev_disp = EventFrameDisplay(producer_name=prod_cd_string, frame_gen_name=frame_gen_string,
                                step=args.step_duration)

    # Now we create the Python Consumer,  adding the callback, and the sources
    pyconsumer = mvd_core.PythonConsumer(ev_disp.event_callback)
    pyconsumer.add_source(frame_gen, frame_gen_string)
    pyconsumer.add_source(cd_producer, prod_cd_string)
    controller.add_component(pyconsumer, "PythonConsumer")

    # initialize figure for the graph plotting
    evrates = []
    times = []
    fig, (events_canvas, rate_canvas) = plt.subplots(
        1, 2, figsize=(16, 6), num="Events & Data Rate (MEv/s) Viewer")
    fig.subplots_adjust(left=0.05, right=0.95)
    rate_canvas.set_xlim(args.time_window, 0)
    rate_canvas.set_ylim(0, args.y_max_evrate)
    rate_canvas.set(xlabel="Seconds", ylabel="Number of events (Millions)")
    events_canvas.set_axis_off()
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(rate_canvas.bbox)

    # video
    if args.frame_dir_output != '':
        if not os.path.exists(args.frame_dir_output):
            os.makedirs(args.frame_dir_output)
        frame_id = 0

    # init data
    points = rate_canvas.plot(times, evrates)[0]
    disp = None

    # Data rate refresh rate
    next_data_frame = args.data_rate_render_period

    # the main loop
    # Note, since the rendering is done using MatPlotLib, the performance is not very good, be patient
    while not controller.is_done():

        # we control the run of the controller, to process the data in output for the graph creation
        controller.run(args.step_duration,
                       controller.get_time() + args.render_period, True)

        # Update event view
        if disp is None:
            disp = events_canvas.imshow(ev_disp.get_frame())
        else:
            disp.set_data(ev_disp.get_frame())

        # Extract data rate plot
        slice_times, slice_rates = ev_disp.pop_rate_series()
        times = times + slice_times
        evrates = evrates + slice_rates
        sec = 1e-6*controller.get_time()

        # Update data rate figure
        if controller.get_time() > next_data_frame:

            # translate data to get only the latest rate information
            tarr = -1*(np.array(times)-sec)

            # update the data rate graph
            points.set_data(tarr, evrates)
            fig.canvas.restore_region(background)
            rate_canvas.draw_artist(points)
            fig.canvas.blit(rate_canvas.bbox)
            next_data_frame = next_data_frame + args.data_rate_render_period

        # save image for video
        if args.frame_dir_output != '':
            plt.savefig(os.path.join(args.frame_dir_output,
                                     'frame_{:05d}'.format(frame_id)))
            frame_id += 1

        plt.pause(0.001)

    if args.frame_dir_output != '':
        fps = 1000000./float(args.render_period)
        print("Frames were saved in '{}'. You can use ffmpeg to generate a video. E.g, 'ffmpeg -framerate {} -i frame_%05d.png output.mp4'".format(args.frame_dir_output, fps))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
