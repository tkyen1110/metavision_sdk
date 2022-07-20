# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
Detection Data Factory
Collection of DataLoaders.
You can add your own here if sequential_dataset is not good fit.
"""
import os
import glob
import json
from functools import partial
from metavision_ml.data.sequential_dataset import SequentialDataLoader
from metavision_ml.data import box_processing as box_api


def get_classes_from_json(dataset_path):
    """
    Read classes for SequentialDataset

    Args:
        dataset_path: path to dataset containing 'label_map_dictionary.json'
    Returns:
        classes
    """
    if os.path.isdir(dataset_path):
        label_map_path = os.path.join(dataset_path, 'label_map_dictionary.json')
    else:
        label_map_path = dataset_path
    with open(label_map_path, "r") as read_file:
        # label_dic is the original dataset dictionary (id -> class name)
        label_dic = json.load(read_file)
        label_dic = {int(str(key)): str(value) for key, value in label_dic.items()}
        size = max(label_dic.keys())
        classes = list(label_dic.values())
    return classes


def setup_psee_classes(label_map_path, wanted_classes=[]):
    """Setups classes and lookup
    for loading.

    Args:
        label_map_path: path to json containining label dictionary
        wanted_classes: if empty, return all available ones except the "empty" frame label.
    """
    all_classes = get_classes_from_json(label_map_path)
    if not wanted_classes:
        classes = all_classes
    else:
        c1 = set(wanted_classes)
        c2 = set(all_classes)
        inter = c1.difference(c2)
        assert len(inter) == 0, "some classes are not part of this dataset: " + str(inter) + " from: " + str(c2)

    classes = [label for label in classes if label != 'empty']
    class_lookup = box_api.create_class_lookup(label_map_path, wanted_classes)
    return classes, class_lookup


def setup_psee_load_labels(label_map_path, num_tbins, wanted_classes=[], min_box_diag_network=30, interpolation=True):
    """Setups Gen1/Gen4 loading labels

    Args:
        label_map_path: path to json containining label dictionary
        wanted_classes: if empty, return all available ones except the "empty" frame label.
        num_tbins: number of time-bins per batch
        min_box_diag_network: minimum box size to keep
    """
    classes, class_lookup = setup_psee_classes(label_map_path, wanted_classes)
    load_boxes_fn = partial(
        box_api.load_boxes,
        num_tbins=num_tbins,
        class_lookup=class_lookup,
        min_box_diag_network=min_box_diag_network)

    return classes, load_boxes_fn


def psee_data(hparams, mode):
    """This corresponds to SequentialDataset.

    Args:
        hparams: params of pytorch lightning module
        mode: section of data "train", "val" or "test"
    """
    file_path = os.path.join(hparams.dataset_path, mode)
    files = glob.glob(file_path + '/*.h5')
    files += glob.glob(file_path + '/*.dat')
    if len(files) == 0:
        raise FileNotFoundError("No '.h5' or files found, please check your dataset path")

    array_dim = (hparams.num_tbins, hparams.in_channels, hparams.height, hparams.width)

    # We remove 'empty' from the classes
    hparams.classes = [label for label in hparams.classes if label != 'empty']

    label_map_path = os.path.join(hparams.dataset_path, 'label_map_dictionary.json')
    class_lookup = box_api.create_class_lookup(
        label_map_path, hparams.classes)

    load_boxes_fn = partial(
        box_api.load_boxes,
        num_tbins=hparams.num_tbins,
        class_lookup=class_lookup,
        min_box_diag_network=hparams.min_box_diag_network)

    dataloader = SequentialDataLoader(
        files,
        hparams.delta_t,
        hparams.preprocess,
        array_dim,
        load_labels=load_boxes_fn,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        padding=mode != "train")

    return dataloader


def get_dataloader(hparams, mode):
    if hparams.dataset_path == 'toy_problem':
        from metavision_ml.data.moving_mnist import MovingMNISTDataset
        return MovingMNISTDataset(tbins=hparams.num_tbins, batch_size=hparams.batch_size, train=mode == 'train',
                                  height=hparams.height, width=hparams.width,
                                  num_workers=hparams.num_workers,
                                  max_frames_per_video=100,
                                  max_frames_per_epoch=hparams.max_frames_per_epoch)
    else:
        return psee_data(hparams, mode)
