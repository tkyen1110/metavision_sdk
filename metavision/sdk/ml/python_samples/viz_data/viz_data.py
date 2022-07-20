# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

# pylint: disable=no-member

"""
Tool to load HDF5 files and labels from events and wrap them using the dataloader class.

This is a simple example to visualize a dataset using the SequentialDataset class
"""

from functools import partial
import os
import sys
import glob
import argparse
import cv2

from skvideo.io import FFmpegWriter
from itertools import islice
from tqdm import tqdm

from metavision_ml.data.sequential_dataset import SequentialDataLoader
from metavision_ml.detection_tracking.display_frame import draw_box_events
from metavision_ml.utils.main_tools import infer_preprocessing
from metavision_ml.detection.data_factory import psee_data, get_classes_from_json


def viz_parser():
    """CLI interface"""
    parser = argparse.ArgumentParser(description='Visualize a precomputed dataset with box labels.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'dataset_path', metavar='dataset-path',
        help='path to a folder containing a train/val/test split and a label_map_dictionary.json')

    parser.add_argument('--detection', action='store_true',
                        help='if enabled, will attempt to show detections if there are any')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--height-width', nargs=2, default=None, type=int,
                        help="if set, downscale the feature tensor to the requested resolution using interpolation."
                        " Note that only power of two of the original resolutions are available.")
    parser.add_argument('--num-tbins', type=int, default=4, help="timesteps per batch for truncated backprop, this "
                        "corresponds to the number of timeslices that are going to be loaded per batch.")
    parser.add_argument('--num-workers', type=int, default=2, help='number of processes using for the data loading.')
    parser.add_argument('--split', default='train', choices=('train', 'test', 'val'), help='split folder to visualize')
    parser.add_argument('--video-output', default='', type=str,
                        help='if set, generate a video, otherwise show with opencv')

    parser.add_argument('--num-batches', default=-1, type=int,
                        help='visualize X batches, if < 0 will stream an entire epoch')
    return parser


def get_paths(paths, ext=".h5"):
    """
    Takes a list of paths and returns them if they exist and end with ext or searches them if they are folders.
    """
    files = []
    for path in paths:
        if os.path.isdir(path):
            folder_files = list(glob.glob(os.path.join(path, "*" + ext)))
            if folder_files:
                files += folder_files
            else:
                print(f"WARNING : no {ext} files found in folder {path}.")
        elif os.path.exists(path) and path.endswith(ext):
            files.append(path)
        else:
            print(f"WARNING : {path} doesn't end with {ext} or isn't a folder.")
    return files


def autocomplete_params(args):
    array_dim, preprocess_function_name, delta_t, max_incr_per_pixel = infer_preprocessing(args)
    if array_dim is None:
        print(f"ERROR: could not infer the preprocessing")
        sys.exit()

    args.in_channels = array_dim[1]
    if args.height_width is None:
        args.height = array_dim[-2]
        args.width = array_dim[-1]

    args.preprocess = preprocess_function_name
    args.delta_t = delta_t
    args.classes = get_classes_from_json(args.__dict__['dataset_path'])
    args.min_box_diag_network = 0
    return args


def viz_data(args):
    if args.detection:
        dataloader = psee_data(args, args.split)
        label_map = ["background"] + args.classes
        viz_labels = partial(draw_box_events, label_map=label_map)
    else:
        paths = [os.path.join(args.dataset_path, args.split)]
        files = get_paths(paths)
        array_dim = (args.num_tbins, args.in_channels, args.height, args.width)
        dataloader = SequentialDataLoader(files, args.delta_t, args.preprocess, array_dim,
                                          batch_size=args.batch_size, num_workers=args.num_workers)
        viz_labels = None
    if args.video_output:
        video_writer = FFmpegWriter(args.video_output, outputdict={'-crf': '20', '-preset': 'veryslow'})
    else:
        cv2.namedWindow('sequential_dataloader', cv2.WINDOW_NORMAL)

    args.num_batches = args.num_batches if args.num_batches > 0 else len(dataloader)
    for frame in tqdm(islice(dataloader.show(viz_labels), args.num_batches), total=args.num_batches):
        if args.video_output:
            video_writer.writeFrame(frame)
        else:
            cv2.imshow('sequential_dataloader', frame[..., ::-1])
            key = cv2.waitKey(1)
            if key == 27:
                break

    if args.video_output:
        video_writer.close()
    else:
        cv2.destroyWindow('sequential_dataloader')


if __name__ == '__main__':
    parser = viz_parser()
    args = parser.parse_args()
    args = autocomplete_params(args)
    viz_data(args)
