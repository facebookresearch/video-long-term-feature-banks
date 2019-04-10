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

"""
Dataset class for Charades. It stores dataset information such as frame paths,
annotations, etc. It also provides a function for generating "minibatch_info",
which consists of the necessary information for constructing a minibatch
(e.g. frame paths, LFB, labels, etc. ). The actual data loading from disk and
construction of data blobs will be performed by charades_data_input.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import logging
import numpy as np
import os
import random

from core.config import config as cfg
import datasets.dataset_helper as dataset_helper


cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger(__name__)


FPS = cfg.CHARADES.FPS
CENTER_CROP_INDEX = 1


class CharadesDataset():

    def __init__(self, split, lfb_infer_only, shift=None, lfb=None, suffix=''):

        self.blobnames = [blobname + suffix for blobname in [
            'data',
            'labels',
            'lfb',
        ]]

        self._split = split
        self._lfb_infer_only = lfb_infer_only
        self._shift = shift

        if lfb_infer_only:
            self._lfb_enabled = False
        else:
            self._lfb_enabled = cfg.LFB.ENABLED

        # This is used for selecting data augmentation.
        self._split_num = int(split == 'train' and not lfb_infer_only)

        self._get_data()

        if self._split == 'train':
            self._sample_rate = cfg.TRAIN.SAMPLE_RATE
            self._video_length = cfg.TRAIN.VIDEO_LENGTH
            self._batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self._sample_rate = cfg.TEST.SAMPLE_RATE
            self._video_length = cfg.TEST.VIDEO_LENGTH
            self._batch_size = cfg.TEST.BATCH_SIZE

        self._seq_len = self._video_length * self._sample_rate

        # We perform 3-crop testing.
        # Thus 30-clip testing == 3 crops * 10 segments.
        self._num_test_segments = cfg.CHARADES.NUM_TEST_CLIPS // 3
        if self._lfb_enabled:
            self._lfb = lfb
            assert len(self._image_paths) == len(self._lfb), \
                'num videos %d != num videos in LFB %d' % (
                    len(self._image_paths), len(self._lfb))

    def get_db_size(self):
        if self._lfb_infer_only:
            return len(self._lfb_frames)
        elif self._split == 'train':
            return len(self._image_paths)
        else:
            return len(self._image_paths) * cfg.CHARADES.NUM_TEST_CLIPS

    def get_minibatch_info(self, indices):
        """
        Given iteration indices, return the necessarry information for
        constructing a minibatch. This will later be used in charades_data_input.py
        to actually load the data and constructing blobs.
        """
        half_len = self._seq_len // 2

        image_paths = []
        labels = []
        spatial_shift_positions = []
        lfb = []

        if not isinstance(indices, list):
            indices = indices.tolist()

        while len(indices) < self._batch_size // cfg.NUM_GPUS:
            indices.append(indices[0])

        for idx in indices:

            # center_idx is the middle frame in a clip.
            if self._lfb_infer_only:
                video_idx, center_idx = self._lfb_frames[idx]
                num_frames = len(self._image_paths[video_idx])
                spatial_shift_positions.append(CENTER_CROP_INDEX)
            else:
                video_idx = idx % self._num_videos
                num_frames = len(self._image_paths[video_idx])
                if self._split == 'train':
                    center_idx = sample_train_idx(num_frames, self._seq_len)
                    spatial_shift_positions.append(None)
                else:
                    # for, e.g., 30-clip testing, multi_clip_idx stands for
                    # (0-left, 0-center, 0-right, ... 9-left, 9-center, 9-right)
                    multi_clip_idx = idx // self._num_videos

                    spatial_shift_positions.append(multi_clip_idx % 3)
                    segment_id = multi_clip_idx // 3

                    center_idx = sample_center_of_segments(
                        segment_id, num_frames, self._num_test_segments, half_len)

            seq = dataset_helper.get_sequence(
                center_idx, half_len, self._sample_rate, num_frames)

            image_paths.append([self._image_paths[video_idx][frame]
                                for frame in seq])
            labels.append(aggregate_labels(
                [self._image_labels[video_idx][frame]
                 for frame in range(seq[0], seq[-1] + 1)]))
            if self._lfb_enabled:
                lfb.append(sample_lfb(video_idx, center_idx, self._lfb))

        split_list = [self._split_num] * len(indices)

        return (image_paths, labels, split_list, spatial_shift_positions, lfb)

    def _get_data(self):
        """Load frame paths and annotations. """

        # Loading frame paths.
        list_filenames = [
            os.path.join(cfg.CHARADES.FRAME_LIST_DIR, filename)
            for filename in (
                cfg.CHARADES.TRAIN_LISTS
                if (self._split == 'train' or cfg.GET_TRAIN_LFB)
                else cfg.CHARADES.TEST_LISTS)
        ]

        (self._image_paths,
         self._image_labels,
         self._video_idx_to_name, _) = dataset_helper.load_image_lists(
            list_filenames)

        if self._split != 'train':
            # Charades is a video-level task.
            self._convert_to_video_level_labels()

        self._num_videos = len(self._image_paths)

        if self._lfb_infer_only:
            self._lfb_frames = get_lfb_frames(self._image_paths)
            logger.info(
                'Inferring LFB from %d clips in %d videos.' % (
                    len(self._lfb_frames), len(self._image_paths)))

        self.print_summary()

    def _convert_to_video_level_labels(self):
        for video_id in range(len(self._image_labels)):
            video_level_labels = aggregate_labels(self._image_labels[video_id])
            for i in range(len(self._image_labels[video_id])):
                self._image_labels[video_id][i] = video_level_labels

    def print_summary(self):
        logger.info("=== Charades dataset summary ===")
        logger.info('Split: {}'.format(self._split))
        logger.info("Use LFB? {}".format(self._lfb_enabled))
        logger.info('Spatial shift position: {}'.format(self._shift))
        logger.info('Number of videos: {}'.format(len(self._image_paths)))
        total_frames = sum(len(video_img_paths)
                           for video_img_paths in self._image_paths)
        logger.info('Number of frames: {}'.format(total_frames))


def sample_train_idx(num_frames, seq_len):
    """Sample training frames."""
    half_len = seq_len // 2
    if num_frames < seq_len:
        center_idx = num_frames // 2
    else:
        center_idx = random.randint(
            half_len, num_frames - half_len)
    return center_idx


def sample_center_of_segments(segment_id, num_frames,
                              num_test_segments, half_len):
    """Sample testing clips to be the center of uniformly split segments."""
    center_idx = int(np.round(
        (float(num_frames) / num_test_segments)
        * (segment_id + 0.5)))

    return center_idx


def aggregate_labels(label_list):
    """Aggregate a sequence of labels."""
    all_labels = []
    for labels in label_list:
        for l in labels:
            all_labels.append(l)
    return list(set(all_labels))


def get_lfb_frames(image_paths):
    """
    Get frames that will be used to construct LFB.
    The frequency is controlled by CHARADES.LFB_CLIPS_PER_SECOND.
    """
    sample_freq = FPS // cfg.CHARADES.LFB_CLIPS_PER_SECOND

    lfb_frames = []
    for video_idx in range(len(image_paths)):
        num_frames = len(image_paths[video_idx])
        for i in range(num_frames):
            if (i + 1) % sample_freq == 0:
                lfb_frames.append((video_idx, i))
    return lfb_frames


def sample_lfb(video_idx, center_idx, lfb):
    """
    Given a video index and the frame index of the center of a clip,
    return the corresponding LFB.
    """
    if len(lfb[video_idx].keys()) == 0:
        assert False

    secs = cfg.LFB.WINDOW_SIZE // cfg.CHARADES.LFB_CLIPS_PER_SECOND
    begin = int(np.round(center_idx - (float(secs) / 2.0 * FPS)))
    end = begin + secs * FPS

    out_lfb = []
    for frame_idx in range(begin, end + 1):
        if frame_idx in lfb[video_idx]:
            if len(out_lfb) < cfg.LFB.WINDOW_SIZE:
                out_lfb.append(lfb[video_idx][frame_idx])

    out_lfb_arr = np.zeros((cfg.LFB.WINDOW_SIZE, cfg.LFB.LFB_DIM))
    k = len(out_lfb)
    if k > 0:
        out_lfb_arr[:k] = np.array(out_lfb)
    else:
        logger.warm('No LFB loaded for video {}.'.format(video_idx))

    return out_lfb_arr
