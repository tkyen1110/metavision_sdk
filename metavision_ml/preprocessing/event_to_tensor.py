# Copyright (c) Prophesee S.A. - All Rights Reserved
#
# Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").
# You may not use this file except in compliance with these License T&C's.
# A copy of these License T&C's is located in the "licensing" folder accompanying this file.

"""
A collection of preprocessing functions to turn a buffer of events into a sequence of 2 dimensional features

This works only for numpy structured arrays representation of events
This makes intensive use of numba http://numba.pydata.org/ , which is awesome.

Examples:
    >>> delta = 100000
    >>> initial_ts = record.current_time
    >>> events = record.load_delta_t(delta)  # load 100 milliseconds worth of events
    >>> events['t'] -= int(initial_ts)  # events timestamp should be reset
    >>> output_array = np.zeros((1, 2, height, width))  # preallocate output array
    >>> histo(events, output_array, delta)
"""
import math
import numpy as np
from numba import njit, jit


@jit(nopython=True)
def _histo(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, reset=True, max_incr_per_pixel=5.):
    """"""
    if reset:
        output_array[...] = 0
    num_tbins = output_array.shape[0]
    dt = float(total_tbins_delta_t) / num_tbins
    ti = 0
    bin_threshold_int = int(math.ceil(dt))  # we convert to int for performance

    k = 1. / (max_incr_per_pixel * 4 ** downsampling_factor) if max_incr_per_pixel is not None else 1

    for i in range(xypt.shape[0]):
        x, y, p, t = xypt['x'][i] >> downsampling_factor, xypt['y'][i] >> downsampling_factor, xypt['p'][i], xypt['t'][i]

        if t >= bin_threshold_int and ti + 1 < num_tbins:

            ti = int(t // dt)
            bin_threshold_int = int(math.ceil((ti + 1) * dt))

        if output_array[ti, p, y, x] + k <= 1:
            output_array[ti, p, y, x] += k
        else:
            if output_array[ti, p, y, x] != 1:
                output_array[ti, p, y, x] = 1


def histo(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, reset=True, max_incr_per_pixel=5.):
    """Computes histogram on a sequence of events.

    Args:
        xypt (events): structured array containing events
        output_array (np.ndarray): Pre allocated numpy array of shape (num_tbins, 2, height, width)
        total_tbins_delta_t (int): Time interval of the extended time slice (with tbins).
        downsampling_factor (int): Parameter used to reduce the spatial dimension of the obtained feature.
            Actually multiply the coordinates by 2**(-downsampling_factor).
        reset (boolean): whether to reset *output_array* to 0 beforehand.
        max_incr_per_pixel: maximum number of increments per pixel (expressed in initial resolution).
    """
    _histo(xypt, output_array, total_tbins_delta_t, downsampling_factor=downsampling_factor,
           reset=reset, max_incr_per_pixel=max_incr_per_pixel)


@njit
def _event_cube(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, split_polarity=True,
                max_incr_per_pixel=5):
    """
    the loop works because reshape does not modify the memory layout
    """
    height, width = output_array.shape[-2:]
    if split_polarity:
        tensor = output_array.reshape(-1, 2, height, width)
    else:
        tensor = output_array.reshape(-1, 1, height, width)

    t0 = 0
    nbins = tensor.shape[0]

    k = 1. / (max_incr_per_pixel * 4 ** downsampling_factor) if max_incr_per_pixel is not None else 1

    for i in range(xypt.shape[0]):
        x, y, p, t = xypt['x'][i] >> downsampling_factor, xypt['y'][i] >> downsampling_factor, xypt['p'][i], xypt['t'][i]

        ti_star = ((t - t0) * nbins / total_tbins_delta_t) - 0.5
        lbin = math.floor(ti_star)
        rbin = lbin + 1

        left_value = max(0, 1 - abs(lbin - ti_star))
        right_value = 1 - left_value

        if not split_polarity:
            pol = 1 if p else -1
            left_value *= pol
            right_value *= pol
            p = 0

        if 0 <= lbin < nbins:
            new_value = tensor[lbin, p, y, x] + left_value * k

            tensor[lbin, p, y, x] = max(-1, min(new_value, 1))
        if rbin < nbins:
            new_value = tensor[rbin, p, y, x] + right_value * k
            tensor[rbin, p, y, x] = max(-1, min(new_value, 1))


def event_cube(xypt, output_array, total_tbins_delta_t, downsampling_factor=0,
               split_polarity=True, reset=True, max_incr_per_pixel=5):
    """
    Takes xypt events within a timeslice of length total_tbins_delta_t and  updates
    an array with shape (num_tbins,num_utbins*2,H,W) microbins are interval that are used in the channels
    [Unsupervised Event-based Learning of Optical Flow, Zhu et al. 2018]

    Note: you should load delta_t * (tbins+1) to avoid artefacts on last timeslice
                                              because the support of the ime bilinear kernel is 2 bins.

    Args:
        xypt (events): structured array containing events
        output_array (np.ndarray): (num_tbins,num_utbins*2,H,W) dtype MUST be floating point!
        total_tbins_delta_t (int): Length in us of the interval in which events are taken
        downsampling_factor: will divide by this power of 2 the event coordinates x & y.
            (WARNING): This is not like in the paper where you should use bilinear weights for downsampling as well.
            A true event-based bilinear resize should contribute to 8 different cells in result tensor.
        split_polarity (bool): whether or not to split polarity into 2 channels or consider p as weight {-1,1}
        reset (bool): reset output_array, in most cases you should put this to True
        max_incr_per_pixel: maximum number of increments per pixel (expressed in initial resolution).
    """
    assert output_array.dtype == np.float32 or output_array.dtype == np.float64
    if reset:
        output_array[...] = 0
    _event_cube(xypt, output_array, total_tbins_delta_t, downsampling_factor=downsampling_factor,
                split_polarity=split_polarity, max_incr_per_pixel=max_incr_per_pixel)


def timesurface(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, reset=True, normed=True,
                **kwargs):
    """
    Computes a linear two channels time surface.
    The dtype of *output_array* must be sufficient to hold values up to *total_tbins_delta_t* without overflow

    Args:
        xypt (events): structured array containing events
        output_array (np.ndarray): Pre allocated numpy array of shape (num_tbins, 2, height, width)
        total_tbins_delta_t (int): Length in us of the interval in which events are taken
        downsampling_factor (int): Parameter used to reduce the spatial dimension of the obtained feature.
            Actually multiply the coordinates by 2**(-downsampling_factor).
        reset (boolean): whether to reset *output_array* to 0 beforehand.
        normed (boolean): if True, scales the timesurface between 0 and 1.

    """
    dtype = output_array.dtype
    assert dtype.itemsize >= 4 or np.issubdtype(dtype, np.floating)
    _ts(xypt, output_array, total_tbins_delta_t, downsampling_factor=downsampling_factor, reset=reset, normed=normed)


@njit
def _ts(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, reset=True, normed=False):
    """
    This implementation uses counters to know in which time bin and micro time bin to write.
    It relies on the chronological nature of events to avoid some divisions.
    """
    num_tbins = output_array.shape[0]
    if reset:
        output_array[...] = 0

    delta_t = float(total_tbins_delta_t) / num_tbins
    tbin = 0

    bin_threshold_int = int(math.ceil(delta_t))

    for i in range(xypt.shape[0]):
        x, y, p, t = xypt['x'][i] >> downsampling_factor, xypt['y'][i] >> downsampling_factor, xypt['p'][i], xypt['t'][i]

        if t >= bin_threshold_int and tbin + 1 < num_tbins:
            tbin = int(t // delta_t)
            bin_threshold_int = int(math.ceil((tbin + 1) * delta_t))
        value = t - tbin * delta_t
        if not 0 <= value < delta_t:  # discard spurious values due to corrupted timestamps.
            continue

        if normed:
            output_array[tbin, p, y, x] = value / delta_t
        else:
            output_array[tbin, p, y, x] = value


def diff(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, reset=True, max_incr_per_pixel=5.):
    """
    Returns the difference of histogram of positive events and negative events.
    It requires *output_array* have a signed dtype.

    Args:
        xypt (events): structured array containing events
        output_array: numpy float32 array [tbins, 1, height, width]
        total_tbins_delta_t: duration of the timeslice
        downsampling_factor (int): Parameter used to reduce the spatial dimension of the obtained feature.
            Actually multiply the coordinates by 2**(-downsampling_factor).
        reset (boolean): whether to reset *output_array* to 0 beforehand.
        max_incr_per_pixel: maximum number of increments per pixel (expressed in initial resolution).
    """
    dtype = output_array.dtype
    assert np.issubdtype(dtype, np.signedinteger) or np.issubdtype(dtype, np.floating)
    _diff(xypt, output_array, total_tbins_delta_t, downsampling_factor, reset, max_incr_per_pixel=max_incr_per_pixel)


@njit
def _diff(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, reset=True, max_incr_per_pixel=5.):
    if reset:
        output_array[...] = 0
    num_tbins = output_array.shape[0]
    dt = float(total_tbins_delta_t) / num_tbins
    ti = 0
    bin_threshold_int = int(math.ceil(dt))  # we convert to int for performance

    k = 1. / (max_incr_per_pixel * 4 ** downsampling_factor) if max_incr_per_pixel is not None else 1

    for i in range(xypt.shape[0]):
        x, y, p, t = xypt['x'][i] >> downsampling_factor, xypt['y'][i] >> downsampling_factor, xypt['p'][i], xypt['t'][i]

        if t >= bin_threshold_int and ti + 1 < num_tbins:
            ti = int(t // dt)
            bin_threshold_int = int(math.ceil((ti + 1) * dt))

        if p:
            if output_array[ti, 0, y, x] + k <= 1:
                output_array[ti, 0, y, x] += k
            else:
                output_array[ti, 0, y, x] = 1
        else:
            if output_array[ti, 0, y, x] - k >= -1:
                output_array[ti, 0, y, x] -= k
            else:
                output_array[ti, 0, y, x] = -1


def multi_channel_timesurface(xypt, output_array, total_tbins_delta_t, downsampling_factor=0, reset=True, **kwargs):
    """Computes a linear two channels time surface n times per delta_t.

    Input channels of the output_array variable must be even. If it is 4 there will be 2 micro delta_t, 3 channels if it
    is 6 and so on. This feature contains precise time information and allows to perceive higher frequency phenomenon
    than 1 / delta_t.
    The dtype of *output_array* must be sufficient to hold values up to *total_tbins_delta_t* without overflow

    Args:
        xypt (events): structured array containing events
        output_array (np.ndarray): (num_tbins,num_utbins*2,H,W) dtype MUST be floating point!
        total_tbins_delta_t (int): length in us of the interval in which events are taken
        downsampling_factor: will divide by this power of 2 the event coordinates x & y.
        reset (bool): whether to reset *output_array* to 0 before computing the feature.
    """
    assert output_array.dtype == np.float32 or output_array.dtype == np.float64, "preprocess_dtype must be float !"
    assert output_array.shape[1] % 2 == 0, "multi_channel_timesurface must have an even number of input channels"

    num_channels, height, width = output_array.shape[-3:]
    tensor = output_array.reshape(-1, 2, height, width)
    _ts(xypt, tensor, total_tbins_delta_t, downsampling_factor=downsampling_factor,
        reset=reset, normed=True)
