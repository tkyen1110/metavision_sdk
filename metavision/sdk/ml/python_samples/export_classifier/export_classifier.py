# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.
"""
Main script for export classification model to Torch.Jit.
"""

import os
import argparse
import torch

from metavision_ml.classification.lightning_model import ClassificationModel
import json
import numpy as np

PARAMS_TO_EXPORT = ["delta_t", "label_delta_t", "use_label_freq", "models", "in_channels", "height", "width",
                    "preprocess", "max_incr_per_pixel"]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def export_classifier(lightning_model, out_directory, tseq, batch_size):
    """Exports Jitted classifier
    & json parameter files
    Args:
        lightning_model : Pytorch lightning classification model
        out_directory: output directory
        tseq (int): time sequence of one random input tensor
        batch_size (int): batch size of one random input tensor
    """

    classifier = lightning_model.net.cpu()
    classifier.eval()
    params = lightning_model.hparams
    label_map = ['background'] + params['classes']

    jit_model = torch.jit.script(classifier)
    jit_model.save(os.path.join(out_directory, "model_classifier.ptjit"))

    # export relevant params for inference
    dic_json = {key: int(params[key]) if type(params[key]).__module__ == np.__name__ else params[key] for key in
                PARAMS_TO_EXPORT}
    dic_json["label_map"] = label_map
    dic_json["num_classes"] = len(label_map)

    filename_json = os.path.join(out_directory, "info_classifier_jit.json")
    json.dump(dic_json, open(filename_json, "w"), indent=4, default=lambda o: o.__dict__, sort_keys=True)

    # sanity check
    x = torch.rand((tseq, batch_size, params['in_channels'], params['height'], params['width']))
    ckpt_out = classifier(x)
    jit_out = jit_model(x)
    np.testing.assert_allclose(to_numpy(ckpt_out), to_numpy(jit_out), rtol=1e-6, atol=1e-6)
    print("torchjit result has been tested, OK")


def main(
        checkpoint_path,
        out_directory,
        tseq=1,
        batch_size=12):
    """
    Performs the export of a model

    Args:
        checkpoint_path (str): path to checkpoint file saved during training
        out_directory (str): output directory where the exported model will be saved
        tseq (int): time sequence of one random input tensor
        batch_size (int): batch size of one random input tensor
    """
    # 1. create directory
    if not os.path.exists(out_directory):
        print('Creating destination folder: {}'.format(out_directory))
        os.makedirs(out_directory)

    # 2. load classification model
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    hparams = argparse.Namespace(**checkpoint['hyper_parameters'])
    model = ClassificationModel(hparams)
    model.load_state_dict(checkpoint['state_dict'])

    # 3. export
    export_classifier(model, out_directory, tseq, batch_size)


if __name__ == '__main__':
    import fire

    fire.Fire(main)
