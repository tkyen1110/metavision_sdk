# Copyright (c) Prophesee S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
6-DOF motion in front of image plane
All in numpy + OpenCV
Applies continuous homographies to your picture in time.
Also you can get the optical flow for this motion.
"""
from __future__ import absolute_import

import numpy as np
import cv2

from metavision_core_ml.data.camera_poses import CameraPoseGenerator


class PlanarMotionStream(object):
    """
    Generates a planar motion in front of the image

    Args:
        image_filename: path to image
        height: desired height
        width: desired width
        max_frames: number of frames to stream
        rgb: color images or gray
        infinite: border is mirrored
    """

    def __init__(self, image_filename, height, width, max_frames=1000, rgb=False, infinite=True, pause_probability=0.5):
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self.rgb = rgb
        self.filename = image_filename
        if not self.rgb:
            frame = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)
        else:
            frame = cv2.imread(image_filename)
        self.frame = frame
        self.frame_height, self.frame_width = self.frame.shape[:2]
        if self.height == -1 or self.width == -1:
            self.height, self.width = self.frame_height, self.frame_width
        self.camera = CameraPoseGenerator(self.frame_height, self.frame_width, self.max_frames, pause_probability)
        self.iter = 0
        self.border_mode = cv2.BORDER_REFLECT101 if infinite else cv2.BORDER_CONSTANT
        self.dt = np.random.randint(10000, 20000)
        self.xy1 = None

    def get_size(self):
        return (self.height, self.width)

    def pos_frame(self):
        return self.iter

    def __len__(self):
        return self.max_frames

    def __next__(self):
        if self.iter >= len(self.camera):
            raise StopIteration

        G_0to2, ts = self.camera()

        out = cv2.warpPerspective(
            self.frame,
            G_0to2,
            dsize=(self.frame_width, self.frame_height),
            borderMode=self.border_mode,
        )
        self.iter += 1
        ts *= self.dt
        out = cv2.resize(out, (self.width, self.height), 0, 0, cv2.INTER_AREA)
        return out, ts

    def __iter__(self):
        return self

    def get_relative_homography(self, time_step):
        rvec1, tvec1 = self.camera.rvecs[time_step], self.camera.tvecs[time_step]
        rvec2, tvec2 = self.camera.rvecs[self.iter-1], self.camera.tvecs[self.iter-1]
        H_2_1 = self.camera.get_transform(rvec2, tvec2, rvec1, tvec1, self.height, self.width)
        return H_2_1
