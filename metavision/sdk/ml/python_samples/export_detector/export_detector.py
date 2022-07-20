# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Main script for export to Torch.Jit.
This allows running your trained model within the C++ Detection & Tracking Pipeline.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import torch

from metavision_ml.detection.jitting import export_lightning_model
from metavision_ml.detection.jitting_test import run_all_tests
from metavision_ml.detection.lightning_model import LightningDetectionModel


def main(
        checkpoint_path,
        out_directory,
        test_height=None,
        test_width=None,
        nms_thresh=0.4,
        score_thresh=0.4,
        verification_sequence="",
        detection_and_tracking_pipeline_script=None):
    """
    Performs the export of a model

    Args:
        checkpoint_path (str): path to checkpoint file saved during training
        out_directory (str): output directory where the exported model will be saved
        test_height (int): height to test the exported model (if a sequence is provided)
        test_width (int): width to test the exported model (if a sequence is provided)
        nms_thresh (float): threshold value for Non-Maximal Suppression of bounding boxes
        score_thresh (float): threshold value of probability to consider a bounding box
        verification_sequence (string): if a recording is provided (optional parameter), we display
                                        the results of the detection and tracking pipeline on this sequence
        detection_and_tracking_pipeline_script (script): path to detection and tracking python script
    """
    # 1. create directory
    if not os.path.exists(out_directory):
        print('Creating destination folder: {}'.format(out_directory))
        os.makedirs(out_directory)

    # 2. load model
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
    model = LightningDetectionModel(hparams)
    model.load_state_dict(checkpoint['state_dict'])

    # 3. export
    export_lightning_model(
        model,
        out_directory,
        nms_thresh=nms_thresh,
        score_thresh=score_thresh)

    # 4. test
    print('Test:')
    run_all_tests(checkpoint_path=checkpoint_path, jit_directory=out_directory,
                  sequence_raw_filename=verification_sequence)

    if detection_and_tracking_pipeline_script and not verification_sequence:
        print("Unable to test the detection_and_tracking_pipeline: please also provide a verification sequence (using --verification_sequence)")

    # 5. run detection and tracking pipeline to display the results visually
    if verification_sequence and detection_and_tracking_pipeline_script:
        assert os.path.isfile(verification_sequence)
        assert os.path.isfile(detection_and_tracking_pipeline_script)
        dt_pipeline_dir = os.path.dirname(detection_and_tracking_pipeline_script)
        print("Computing detection and tracking on sequence: ", verification_sequence)
        print("type 'Q' to quit")
        assert os.path.isdir(dt_pipeline_dir)
        sys.path.append(dt_pipeline_dir)
        import detection_and_tracking_pipeline as dtp

        model_dir = out_directory

        args_list = []
        args_list += ['--record_file', verification_sequence]
        args_list += ['--object_detector_dir', model_dir]
        args_list += ['--detector_confidence_threshold', str(score_thresh)]
        if test_width is not None:
            args_list += ['--network_input_width', str(test_width)]
        if test_height is not None:
            args_list += ['--network_input_height', str(test_height)]
        args_list += ['--display']
        if not torch.cuda.is_available():
            args_list += ['--device', 'cpu']
        args = dtp.parse_args(args_list)
        dtp.run(args)


if __name__ == '__main__':
    import fire
    fire.Fire(main)
