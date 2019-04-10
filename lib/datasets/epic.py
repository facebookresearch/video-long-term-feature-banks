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
Dataset class for EPIC-Kitchens. It stores dataset information such as frame paths,
annotations, etc. It also provides a function for generating "minibatch_info",
which consists of the necessary information for constructing a minibatch
(e.g. frame paths, LFB, labels, etc. ). The actual data loading from disk and
construction of data blobs will be performed by epic_data_input.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
import cv2
import logging
import numpy as np
import os
import random

from core.config import config as cfg
import datasets.dataset_helper as dataset_helper


cv2.ocl.setUseOpenCL(False)
logger = logging.getLogger(__name__)


FPS = cfg.EPIC.FPS
CENTER_CROP_INDEX = 1
TRAIN_PERSON_INDICES = range(1, 26)
NUM_CLASSES_VERB = 125
NUM_CLASSES_NOUN = 352


class EpicDataset():

    def __init__(self, split, lfb_infer_only, shift=None, lfb=None, suffix=''):

        self.blobnames = [blobname + suffix for blobname in [
            'data',
            'labels',
            'lfb',
        ]]

        self._split = split
        self._is_train = (self._split == 'train')
        self._lfb_infer_only = lfb_infer_only
        self._shift = shift

        if lfb_infer_only:
            self._lfb_enabled = False
        else:
            self._lfb_enabled = cfg.LFB.ENABLED

        # This is used for selecting data augmentation.
        self._split_num = int(self._is_train and not lfb_infer_only)

        self._get_data()

        if self._is_train:
            self._sample_rate = cfg.TRAIN.SAMPLE_RATE
            self._video_length = cfg.TRAIN.VIDEO_LENGTH
            self._batch_size = cfg.TRAIN.BATCH_SIZE
        else:
            self._sample_rate = cfg.TEST.SAMPLE_RATE
            self._video_length = cfg.TEST.VIDEO_LENGTH
            self._batch_size = cfg.TEST.BATCH_SIZE

        self._seq_len = self._video_length * self._sample_rate

        if self._lfb_enabled:
            self._lfb = lfb

            assert len(self._image_paths) == len(self._lfb), \
                'num videos %d != num videos in LFB %d' % (
                    len(self._image_paths), len(self._lfb))

    def get_db_size(self):
        return len(self._annotations)

    def get_minibatch_info(self, indices):
        """
        Given iteration indices, return the necessarry information for
        constructing a minibatch. This will later be used in epic_data_input.py
        to actually load the data and constructing blobs.
        """

        half_len = self._seq_len // 2

        image_paths = []
        labels = []
        lfb = []

        if not isinstance(indices, list):
            indices = indices.tolist()

        while len(indices) < self._batch_size // cfg.NUM_GPUS:
            indices.append(indices[0])

        for idx in indices:

            if self._is_train:
                ann_idx = np.random.choice(range(len(self._annotations)))
            else:
                ann_idx = idx

            (person, video_name,
             start_frame, stop_frame, verb, noun) = self._annotations[ann_idx]

            num_frames = len(self._image_paths[video_name])

            seq, center_idx = get_sequence(
                start_frame, stop_frame, half_len, self._sample_rate,
                num_frames, self._is_train)

            image_paths.append(
                [self._image_paths[video_name][frame] for frame in seq])

            labels.append(verb if cfg.EPIC.CLASS_TYPE == 'verb' else noun)
            if self._lfb_enabled:
                lfb.append(self.sample_lfb(video_name, center_idx))

        split_list = [self._split_num] * len(indices)

        if self._is_train:
            spatial_shift_positions = [None] * len(indices)
        elif self._shift is None:
            spatial_shift_positions = [CENTER_CROP_INDEX] * len(indices)
        else:
            spatial_shift_positions = [self._shift] * len(indices)

        return (image_paths, labels, split_list, spatial_shift_positions, lfb)

    def _get_data(self):
        """Load frame paths and annotations. """

        # Load frame paths.
        list_filenames = [
            os.path.join(cfg.EPIC.FRAME_LIST_DIR, filename)
            for filename in (
                cfg.EPIC.TRAIN_LISTS
                if (self._is_train or cfg.GET_TRAIN_LFB)
                else cfg.EPIC.TEST_LISTS)
        ]

        (self._image_paths,
         self._image_labels,
         self._video_idx_to_name,
         self._video_name_to_idx) = dataset_helper.load_image_lists(
            list_filenames, return_dict=True)

        # Load annotations.
        if self._lfb_infer_only:
            self._annotations = get_annotations_for_lfb_frames(
                self._image_paths)
            logger.info(
                'Inferring LFB from %d clips in %d videos.' % (
                    len(self._annotations), len(self._image_paths)))
        else:
            self._annotations = load_annotations(is_train=self._is_train)

        self.print_summary()

    def print_summary(self):
        logger.info("=== EPIC Kitchens dataset summary ===")
        logger.info('Split: {}'.format(self._split))
        logger.info("Use LFB? {}".format(self._lfb_enabled))
        logger.info('Spatial shift position: {}'.format(self._shift))
        logger.info('Number of videos: {}'.format(len(self._image_paths)))
        total_frames = sum(len(video_img_paths)
                           for video_img_paths in self._image_paths.values())
        logger.info('Number of frames: {}'.format(total_frames))
        logger.info('Number of annotations: {}'.format(len(self._annotations)))

    def sample_lfb(self, video_name, center_idx):
        """Sample LFB. Note that for verbs, we use video-model based LFB, and
        for nouns, we use detector-based LFB, so the formats are slightly
        different. Thus we use different functions for verb LFB and noun LFB."""
        if cfg.EPIC.CLASS_TYPE == 'noun':
            return sample_noun_lfb(
                center_idx, self._lfb[self._video_name_to_idx[video_name]])
        else:
            return sample_verb_lfb(center_idx, self._lfb[video_name])


def sec_to_frame(sec):
    """Time index (in seconds) to frame index."""
    return int(np.round(float(sec) * FPS))


def frame_to_sec(frame):
    """Frame index to time index (in seconds)."""
    return int(np.round(float(frame) / FPS))


def time_to_sec(sec):
    """Parse time string. (e.g. "00:02:10.99")"""
    hour, minute, sec = sec.split(':')
    return 3600.0 * int(hour) + 60.0 * int(minute) + float(sec)


def get_sequence(start_frame, stop_frame, half_len,
                    sample_rate, num_frames, is_train):
    """Get a sequence of frames (for a clip) with appropriete padding."""
    if is_train:
        center_frame = random.randint(start_frame, stop_frame)
    else:
        center_frame = (stop_frame + start_frame) // 2
    seq = range(center_frame - half_len, center_frame + half_len, sample_rate)

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1

    return seq, center_frame


def load_annotations(is_train):
    """Load EPIC-Kitchens annotations."""
    annotations = []

    verb_set = set()
    noun_set = set()

    filename = os.path.join(cfg.EPIC.ANNOTATION_DIR, cfg.EPIC.ANNOTATIONS)
    with open(filename, 'rb') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            person = row[1]

            if is_train:
                if int(person[1:]) not in TRAIN_PERSON_INDICES:
                    continue
            else:
                if int(person[1:]) in TRAIN_PERSON_INDICES:
                    continue

            #uid,participant_id,video_id,narration,start_timestamp,stop_timestamp,start_frame,stop_frame,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
            video_name = row[2]
            start_frame = sec_to_frame(time_to_sec(row[4]))
            stop_frame = sec_to_frame(time_to_sec(row[5]))
            verb = int(row[-5])
            noun = int(row[-3])

            assert verb < NUM_CLASSES_VERB, verb
            assert verb >= 0, verb
            assert noun < NUM_CLASSES_NOUN, noun
            assert noun >= 0, noun

            annotations.append(
                (person, video_name, start_frame, stop_frame, verb, noun))
            verb_set.add(verb)
            noun_set.add(noun)

    logger.info('See %d verbs and %d nouns in the dataset loaded.' % (
        len(verb_set), len(noun_set)))

    cur_label_set = verb_set if cfg.EPIC.CLASS_TYPE == 'verb' else noun_set
    if len(cur_label_set) != cfg.MODEL.NUM_CLASSES:
        logger.warn(
            '# classes seen (%d) != MODEL.NUM_CLASSES' % len(cur_label_set))
    assert len(annotations) == (cfg.TRAIN.DATASET_SIZE if is_train
                                else cfg.TEST.DATASET_SIZE)
    return annotations


def get_annotations_for_lfb_frames(image_paths):
    """
    Return the "annotations" that correspond to the frames/clips that will be
    used to construct LFB. The sampling is done uniformly with frequency
    controlled by EPIC.VERB_LFB_CLIPS_PER_SECOND.
    """
    annotations = []

    sample_freq = FPS // cfg.EPIC.VERB_LFB_CLIPS_PER_SECOND

    for video_name in image_paths.keys():
        for img_path in image_paths[video_name]:

            frame = filename_to_frame_id(img_path)
            if frame % sample_freq == 0:
                annotations.append((video_name[:3], video_name, frame, frame, 0, 0))

    return annotations


def filename_to_frame_id(img_path):
    return int(img_path[-10:-4])


def sample_verb_lfb(center_idx, video_lfb):
    """Sample verb LFB."""
    window_size = cfg.LFB.WINDOW_SIZE
    half_len = (window_size * FPS) // 2

    lower = center_idx - half_len
    upper = center_idx + half_len

    out_lfb = []
    for frame_idx in range(lower, upper + 1):
        if frame_idx in video_lfb.keys():
            if len(out_lfb) < window_size:
                out_lfb.append(video_lfb[frame_idx])

    out_lfb = np.array(out_lfb)
    if out_lfb.shape[0] < window_size:
        new_out_lfb = np.zeros((window_size, cfg.LFB.LFB_DIM))
        if out_lfb.shape[0] > 0:
            new_out_lfb[:out_lfb.shape[0]] = out_lfb
        out_lfb = new_out_lfb

    return out_lfb.astype(np.float32)


def is_empty_list(x):
    return isinstance(x, (list,)) and len(x) == 0


def sample_noun_lfb(center_idx, video_lfb):
    """Sample noun LFB."""
    max_num_feat_per_frame = cfg.EPIC.MAX_NUM_FEATS_PER_NOUN_LFB_FRAME
    window_size = cfg.LFB.WINDOW_SIZE

    secs = float(window_size) / (max_num_feat_per_frame
                                 * cfg.EPIC.NOUN_LFB_FRAMES_PER_SECOND)
    lower = int(center_idx - (secs / 2) * FPS)
    upper = int(lower + secs * FPS)

    out_lfb = []
    num_feat = 0
    for frame_idx in range(lower, upper + 1):
        if frame_idx in video_lfb:
            frame_lfb = video_lfb[frame_idx]
            if not is_empty_list(frame_lfb):
                curr_num = min(max_num_feat_per_frame, frame_lfb.shape[0])
                num_feat += curr_num

                out_lfb.append(frame_lfb[:curr_num])
                if num_feat >= window_size:
                    break

    if len(out_lfb) == 0:
        logger.warn('No LFB sampled (certer_idx: %d)' % center_idx)
        return np.zeros((window_size, cfg.LFB.LFB_DIM))

    out_lfb = np.vstack(out_lfb)[:window_size].astype(np.float32)

    if random.random() < 0.001:
        logger.info(out_lfb.shape)
    if out_lfb.shape[0] < window_size:
        new_out_lfb = np.zeros((window_size, cfg.LFB.LFB_DIM))
        new_out_lfb[:out_lfb.shape[0]] = out_lfb
        out_lfb = new_out_lfb

    return out_lfb
