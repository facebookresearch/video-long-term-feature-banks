# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Herper functions for dataset loading."""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from collections import defaultdict
from core.config import config as cfg
import cv2
import logging
import os


cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger(__name__)


def load_image_lists(list_filenames, return_dict=False):
    image_paths = defaultdict(list)
    labels = defaultdict(list)

    video_name_to_idx = {}
    video_idx_to_name = {}
    for list_filename in list_filenames:
        with open(list_filename, 'r') as f:
            f.readline()
            for line in f:
                row = line.split()
                # original_vido_id video_id frame_id path labels
                assert len(row) == 5
                video_name = row[0]

                if video_name not in video_name_to_idx:
                    idx = len(video_name_to_idx)
                    video_name_to_idx[video_name] = idx
                    video_idx_to_name[idx] = video_name

                if return_dict:
                    data_key = video_name
                else:
                    data_key = video_name_to_idx[video_name]

                image_paths[data_key].append(os.path.join(cfg.DATADIR, row[3]))

                frame_labels = row[-1].replace('\"', '')
                if frame_labels != "":
                    labels[data_key].append(map(int, frame_labels.split(',')))
                else:
                    labels[data_key].append([])

    if return_dict:
        image_paths = dict(image_paths)
        labels = dict(labels)
    else:
        image_paths = [image_paths[i] for i in range(len(image_paths))]
        labels = [labels[i] for i in range(len(labels))]
    return image_paths, labels, video_idx_to_name, video_name_to_idx


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    seq = range(center_idx - half_len, center_idx + half_len, sample_rate)

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq
