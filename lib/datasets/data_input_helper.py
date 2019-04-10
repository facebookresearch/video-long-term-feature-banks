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

"""Helper functions for data input processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from multiprocessing import Pool
from multiprocessing.sharedctypes import RawArray
import collections
import ctypes
import cv2
import logging
import numpy as np
import time

from core.config import config as cfg
import datasets.image_processor as imgproc


logger = logging.getLogger(__name__)


# Mean and Std are BGR based.
DATA_MEAN = np.array(cfg.DATA_MEAN, dtype=np.float32)
DATA_STD = np.array(cfg.DATA_STD, dtype=np.float32)


# PCA is RGB based.
PCA = {
    'eigval': np.array(cfg.TRAIN.PCA_EIGVAL).astype(np.float32),
    'eigvec': np.array(cfg.TRAIN.PCA_EIGVEC).astype(np.float32)
}


def retry_load_images(image_paths, retry):
    for i in range(retry):
        imgs = [cv2.imread(image_path) for image_path in image_paths]

        if all(img is not None for img in imgs):
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            assert False, 'Failed to load images {}'.format(image_paths)


def convert_to_batch(data):
    data = np.concatenate(
        [arr[np.newaxis] for arr in data]).astype(np.float32)
    return np.ascontiguousarray(data)


def images_and_boxes_preprocessing(
        imgs, split, crop_size, spatial_shift_pos, boxes=None):

    height, width, _ = imgs[0].shape

    if boxes is not None:
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height

        boxes = imgproc.clip_boxes_to_image(boxes, height, width)

    # Now the image is in HWC, BGR format
    if split == 1:  # "train"
        imgs, boxes = imgproc.random_short_side_scale_jitter_list(
            imgs,
            min_size=cfg.TRAIN.JITTER_SCALES[0],
            max_size=cfg.TRAIN.JITTER_SCALES[1],
            boxes=boxes,
        )
        imgs, boxes = imgproc.random_crop_list(
            imgs, crop_size, order='HWC', boxes=boxes)

        # random flip
        imgs, boxes = imgproc.horizontal_flip_list(
            0.5, imgs, order='HWC', boxes=boxes)
    else:
        # Short side to cfg.TEST_SCALE. Non-local and STRG uses 256.
        imgs = [imgproc.scale(cfg.TEST.SCALE, img) for img in imgs]
        if boxes is not None:
            boxes = imgproc.scale_boxes(cfg.TEST.SCALE, boxes, height, width)

        if cfg.AVA.FORCE_TEST_FLIP and cfg.DATASET == 'ava':
            imgs, boxes = imgproc.horizontal_flip_list(
                0.5, imgs, order='HWC', boxes=boxes,
                force_flip=True)

        # For the short side we do center crop.
        imgs, boxes = imgproc.spatial_shift_crop_list(
            crop_size, imgs, spatial_shift_pos, boxes=boxes)

    # Convert image to CHW keeping BGR order
    imgs = [imgproc.HWC2CHW(img) for img in imgs]

    # image [0, 255] -> [0, 1]
    imgs = [img / 255.0 for img in imgs]

    imgs = [np.ascontiguousarray(
        img.reshape((3, crop_size, crop_size))).astype(np.float32)
        for img in imgs]

    # do color augmentation (after divided by 255.0)
    if cfg.TRAIN.USE_COLOR_AUGMENTATION and split == 1:
        imgs = color_augmentation_list(imgs)

    # now, normalize by mean and std
    imgs = [imgproc.color_normalization(img, DATA_MEAN, DATA_STD)
            for img in imgs]

    # 3, 224, 224 -> 3, 32, 224, 224
    imgs = np.concatenate(
        [np.expand_dims(img, axis=1) for img in imgs], axis=1)

    # Kinetics pre-training uses RGB!!
    if not cfg.MODEL.USE_BGR:
        # BGR to RGB.
        imgs = imgs[::-1, ...]

    if boxes is not None:
        boxes = imgproc.clip_boxes_to_image(boxes, crop_size, crop_size)
    return imgs, boxes


def color_augmentation_list(imgs):
    if not cfg.TRAIN.PCA_JITTER_ONLY:
        imgs = imgproc.color_jitter_list(
            imgs, img_brightness=0.4, img_contrast=0.4, img_saturation=0.4)

    imgs = imgproc.lighting_list(
        imgs, alphastd=0.1, eigval=PCA['eigval'],
        eigvec=np.array(PCA['eigvec']).astype(np.float32)
    )
    return imgs


def _create_execution_context(execution_context, init_pool, worker_ids, expected_data_size,
                              num_processes, batch_size):

    logger.info('CREATING EXECUTION CONTEXT')
    if execution_context is None:
        pools = {}
        shared_data_lists = {}
    else:
        pools = execution_context.pools
        shared_data_lists = execution_context.shared_data_lists
    logger.info('POOLS: {}'.format(pools))
    logger.info('SHARED DATA LISTS: {}'.format(len(shared_data_lists)))

    if cfg.TRAIN.CROP_SIZE == cfg.TEST.CROP_SIZE:
        scales = [cfg.TRAIN.CROP_SIZE]
    else:
        scales = [cfg.TRAIN.CROP_SIZE, cfg.TEST.CROP_SIZE]

    for worker_id in worker_ids:
        # for each worker_id, create a shared pool
        shared_data_list = [[] for i in range(len(scales))]
        shared_data_lists[worker_id] = shared_data_list
        logger.info('worker_id: {} list: {}'.format(
            worker_id, len(shared_data_lists)))
        logger.info('worker_id: {} list keys: {}'.format(
            worker_id, shared_data_lists.keys()))
        # for each worker_id, we fetch a batch size of 32 and this is being
        # done by various parallel processes
        for i in range(len(scales)):
            if scales[i] == cfg.TRAIN.CROP_SIZE:
                bz = cfg.TRAIN.BATCH_SIZE
            else:
                bz = cfg.TEST.BATCH_SIZE
            for _ in range(bz):
                shared_arr = RawArray(
                    ctypes.c_float,
                    scales[i] ** 2 * 3 * cfg.TRAIN.VIDEO_LENGTH)

                one_data_list = [shared_arr]
                if cfg.DATASET == 'ava':
                    shared_arr_box = RawArray(
                        ctypes.c_float,
                        cfg.LFB.NUM_LFB_FEAT * 4)

                    shared_arr_original_boxes = RawArray(
                        ctypes.c_float,
                        cfg.LFB.NUM_LFB_FEAT * 4)

                    # height, width
                    shared_arr_metadata = RawArray(
                        ctypes.c_float,
                        2)
                    one_data_list += [
                        shared_arr_box,
                        shared_arr_original_boxes,
                        shared_arr_metadata]

                shared_data_list[i].append(one_data_list)

        pools[worker_id] = Pool(
            processes=num_processes,
            initializer=init_pool,
            initargs=(shared_data_list,)
        )
    context = collections.namedtuple(
        'ExecutionContext',
        ['pools', 'shared_data_lists']
    )
    context.pools = pools
    context.shared_data_lists = shared_data_lists
    logger.info('CREATED POOL: {}'.format(pools))
    logger.info('CREATED LISTS: {}'.format(len(shared_data_lists)))
    logger.info('POOL keys: {}'.format(pools.keys()))
    logger.info('LIST keys: {}'.format(shared_data_lists.keys()))
    return context


def _shutdown_pools(pools):
    logger.info("Shutting down multiprocessing pools..")
    for i, p in enumerate(pools.values()):
        logger.info("Shutting down pool {}".format(i))
        try:
            p.close()
            p.join()
        except Exception as e:
            logger.info(e)
            continue
    logger.info("Pools closed")
