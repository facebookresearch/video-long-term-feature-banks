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
Multi-process data loading for Charades.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import atexit
import logging
import numpy as np

from core.config import config as cfg
import datasets.data_input_helper as data_input_helper


logger = logging.getLogger(__name__)
execution_context = None


def create_data_input(
    input_db, expected_data_size, num_processes, num_workers, split,
    batch_size, crop_size=cfg.TRAIN.CROP_SIZE,
):
    # create a global execution context for the dataloader which contains the
    # pool for each thread and each pool has num_processes and a shared data list
    global execution_context

    def init(worker_ids):
        global execution_context

        logging.info('Creating the execution context for '
            'worker_ids: {}, batch size: {}'.format(
                worker_ids,
                batch_size))

        execution_context = data_input_helper._create_execution_context(
            execution_context, _init_pool, worker_ids, expected_data_size,
            num_processes, batch_size)

        atexit.register(_shutdown_pools)

    # in order to get the minibatch, we need some information from the db class
    def get_minibatch_out(
            input_db, worker_id, batch_size, db_indices, crop_size):
        """
        Get minibatch info from CharadesDataset and perform the actual
        minibatch loading.
        """
        pools = execution_context.pools
        shared_data_lists = execution_context.shared_data_lists
        curr_pool = pools[worker_id]
        shared_data_list = shared_data_lists[worker_id]

        minibatch_info = input_db.get_minibatch_info(db_indices)

        return _load_and_process_images(
            worker_id, curr_pool, shared_data_list, crop_size,
            minibatch_info, input_db)

    return (init, get_minibatch_out)


def construct_label_array(video_labels):
    """Construction label array."""
    label_arr = np.zeros((cfg.MODEL.NUM_CLASSES, ))

    for lbl in set(video_labels):
        label_arr[lbl] = 1
    return label_arr.astype(np.int32)


def _load_and_process_images(
    worker_id, curr_pool, shared_data_list, crop_size, minibatch_info, input_db
):
    """Construct a minibatch given minibatch_info."""

    (image_paths, labels, split_list,
     spatial_shift_positions, lfb) = minibatch_info

    if crop_size == cfg.TEST.CROP_SIZE:
        curr_shared_list_id = len(shared_data_list) - 1
    else:
        curr_shared_list_id = 0

    map_results = curr_pool.map_async(
        get_clip_from_source,
        zip(
            [i for i in range(0, len(image_paths))],
            image_paths,
            split_list,
            [crop_size for i in range(0, len(image_paths))],
            spatial_shift_positions,
            [curr_shared_list_id for i in range(0, len(image_paths))],
        )
    )

    out_images = []
    out_labels = []
    for index in map_results.get():
        if index is not None:
            np_arr, = shared_data_list[curr_shared_list_id][index]
            tmp_np_arr = np.reshape(
                np_arr, (3, cfg.TRAIN.VIDEO_LENGTH, crop_size, crop_size))
            out_images.append(tmp_np_arr)
            out_labels.append(labels[index])

    out_images = data_input_helper.convert_to_batch(out_images)
    out_labels = np.array([construct_label_array(lbl) for lbl in out_labels])
    out_lfb = np.array(lfb).astype(np.float32)

    return (out_images, out_labels, out_lfb)


def get_clip_from_source(args):
    (index, image_paths, split, crop_size, spatial_shift_pos, list_id) = args
    """Load images/data from disk and pre-process data."""

    try:
        imgs = data_input_helper.retry_load_images(image_paths,
                                                   cfg.IMG_LOAD_RETRY)

        imgs, _ = data_input_helper.images_and_boxes_preprocessing(
            imgs, split, crop_size, spatial_shift_pos)

        np_arr = shared_data_list[list_id][index][0]
        np_arr = np.reshape(np_arr, imgs.shape)

        np_arr[:] = imgs

    except Exception as e:
        logger.error('get_image_from_source failed: '
                     '(index, image_path, split): {} {} {}'.format
                     (index, image_paths, split))
        logger.info(e)
        return None
    return index


def _init_pool(data_list):
    """
    Each pool process calls this initializer.
    Load the array to be populated into that process's global namespace.
    """
    global shared_data_list
    shared_data_list = data_list


def _shutdown_pools():
    data_input_helper._shutdown_pools(execution_context.pools)
