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
#
# Based on:
# Copyright (c) 2017-present, Facebook, Inc.
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
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from utils.collections import AttrDict

logger = logging.getLogger(__name__)


__C = AttrDict()
config = __C

__C.DEBUG = False

__C.DATALOADER = AttrDict()
__C.DATALOADER.MAX_BAD_IMAGES = 100

__C.DATA_MEAN = [0.45, 0.45, 0.45]
__C.DATA_STD = [0.225, 0.225, 0.225]

# Training options.
__C.TRAIN = AttrDict()
__C.TRAIN.PARAMS_FILE = b''
__C.TRAIN.DATA_TYPE = b'train'
__C.TRAIN.BATCH_SIZE = 64

# If the pre-training batchsize does not match the current.
__C.TRAIN.RESUME_FROM_BATCH_SIZE = -1
__C.TRAIN.RESET_START_ITER = False

# Data augmeantion.
__C.TRAIN.JITTER_SCALES = [256, 480]
__C.TRAIN.CROP_SIZE = 224
__C.TRAIN.USE_COLOR_AUGMENTATION = False
# PCA is RGB based
__C.TRAIN.PCA_EIGVAL = [0.225, 0.224, 0.229]
__C.TRAIN.PCA_EIGVEC = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203]]

__C.TRAIN.COMPUTE_PRECISE_BN = True
__C.TRAIN.ITER_COMPUTE_PRECISE_BN = 200

# Number of iterations after which model should be tested on test/val data.
__C.TRAIN.EVAL_PERIOD = 4000
__C.TRAIN.DATASET_SIZE = 0

# Number of frames per clip.
__C.TRAIN.VIDEO_LENGTH = 32
# We sample one frame every "SAMPLE_RATE" frames to form a clip.
__C.TRAIN.SAMPLE_RATE = 2
__C.TRAIN.DROPOUT_RATE = 0.0

__C.TRAIN.TEST_AFTER_TRAIN = True

# Train model options.
__C.MODEL = AttrDict()
__C.MODEL.NUM_CLASSES = -1
__C.MODEL.MODEL_NAME = b''

# 1: C2D, ResNet50, 2: I3D, ResNet50, 3: C2D, ResNet101, 4: I3D, ResNet101.
__C.MODEL.VIDEO_ARC_CHOICE = 2

# Number of ResNet layers.
__C.MODEL.DEPTH = 50

# Batch norm related options.
__C.MODEL.BN_MOMENTUM = 0.9
__C.MODEL.BN_EPSILON = 1.0000001e-5

# We may use 0 to initialize the residual branch of a residual block,
# so the inital state of the block is exactly identiy. This helps optimizaiton.
__C.MODEL.BN_INIT_GAMMA = 1.0

__C.MODEL.FC_INIT_STD = 0.01

__C.MODEL.MEAN = 114.75
__C.MODEL.STD = 57.375

# Options to optimize memory usage.
__C.MODEL.ALLOW_INPLACE_SUM = True
__C.MODEL.ALLOW_INPLACE_RELU = True
__C.MODEL.ALLOW_INPLACE_RESHAPE = True
__C.MODEL.MEMONGER = True

__C.MODEL.USE_BGR = False  # Default is False for historical reasons.

# When fine-tuning with BN frozen, we turn a BN layer into an affine layer.
__C.MODEL.USE_AFFINE = False

__C.MODEL.SAMPLE_THREADS = 8

# AVA & Charades: True. EPIC-Kitchens: False (classification).
__C.MODEL.MULTI_LABEL = True
__C.MODEL.DILATIONS_AFTER_CONV5 = True
__C.MODEL.FREEZE_BACKBONE = False

# For ResNet or ResNeXt.
__C.RESNETS = AttrDict()
__C.RESNETS.NUM_GROUPS = 1
__C.RESNETS.WIDTH_PER_GROUP = 64
__C.RESNETS.STRIDE_1X1 = False
__C.RESNETS.TRANS_FUNC = b'bottleneck_transformation'

__C.TEST = AttrDict()
__C.TEST.PARAMS_FILE = b''
__C.TEST.DATA_TYPE = b''
__C.TEST.BATCH_SIZE = 64
__C.TEST.SCALE = 256
__C.TEST.CROP_SIZE = 256

__C.TEST.DATASET_SIZE = 0

# Number of frames per clip.
__C.TEST.VIDEO_LENGTH = 32
# We sample one frame every "SAMPLE_RATE" frames to form a clip.
__C.TEST.SAMPLE_RATE = 2

# 0: left, 1: center, 2: right.
__C.TEST.CROP_SHIFT = 1

__C.SOLVER = AttrDict()
__C.SOLVER.NESTEROV = True
__C.SOLVER.WEIGHT_DECAY = 0.0001
__C.SOLVER.WEIGHT_DECAY_BN = 0.0001
__C.SOLVER.MOMENTUM = 0.9

# Learning rates
__C.SOLVER.LR_POLICY = b'steps_with_relative_lrs'
__C.SOLVER.BASE_LR = 0.1

__C.SOLVER.STEP_SIZES = [100000, 20000, 20000]
__C.SOLVER.LRS = [1, 0.1, 0.01]
__C.SOLVER.MAX_ITER = 140000

# To be consistent with Detectron, we will turn STEP_SIZES into STEPS.
# Example: STEP_SIZES [30, 30, 20] => STEPS [0, 30, 60, 80].
__C.SOLVER.STEPS = None
__C.SOLVER.GAMMA = 0.1  # For cfg.SOLVER.LR_POLICY = 'steps_with_decay'.

__C.SOLVER.SCALE_MOMENTUM = False

# Only apply the correction if the relative LR change exceeds this threshold
# (prevents ever change in linear warm up from scaling the momentum by a tiny
# amount; momentum scaling is only important if the LR change is large.)
__C.SOLVER.SCALE_MOMENTUM_THRESHOLD = 1.1

__C.SOLVER.WARMUP = AttrDict()
__C.SOLVER.WARMUP.WARMUP_ON = False
__C.SOLVER.WARMUP.WARMUP_START_LR = 0.1
__C.SOLVER.WARMUP.WARMUP_END_ITER = 5000

__C.CHECKPOINT = AttrDict()
__C.CHECKPOINT.CHECKPOINT_MODEL = True
__C.CHECKPOINT.CHECKPOINT_PERIOD = -1
__C.CHECKPOINT.RESUME = True
__C.CHECKPOINT.DIR = b'.'

# If a pre-trained model is trained with BN layers, and we want to finetune it
# with BN frozen, set this to be True to convert pre-trained BN layers into
# Affine layers.
__C.CHECKPOINT.CONVERT_MODEL = False

__C.NONLOCAL = AttrDict()
__C.NONLOCAL.CONV_INIT_STD = 0.01
__C.NONLOCAL.NO_BIAS = 0
__C.NONLOCAL.USE_MAXPOOL = True
__C.NONLOCAL.USE_SOFTMAX = True
__C.NONLOCAL.USE_ZERO_INIT_CONV = False
__C.NONLOCAL.USE_BN = True
__C.NONLOCAL.USE_SCALE = True
__C.NONLOCAL.USE_AFFINE = False

__C.NONLOCAL.BN_MOMENTUM = 0.9
__C.NONLOCAL.BN_EPSILON = 1.0000001e-5
__C.NONLOCAL.BN_INIT_GAMMA = 0.0

__C.NONLOCAL.LAYER_MOD = 2
__C.NONLOCAL.CONV3_NONLOCAL = True
__C.NONLOCAL.CONV4_NONLOCAL = True

# Others.
__C.DATADIR = b''
__C.DATASET = b''
__C.ROOT_GPU_ID = 0
__C.NUM_GPUS = 8
__C.CUDNN_WORKSPACE_LIMIT = 256
__C.RNG_SEED = 2
__C.USE_CYTHON = False

__C.LOG_PERIOD = 10

__C.PROF_DAG = False

__C.INTERPOLATION = b'INTER_LINEAR'

__C.MINIBATCH_QUEUE_SIZE = 64

__C.AVA = AttrDict()
__C.AVA.FRAME_LIST_DIR = b'data/ava/frame_lists'
__C.AVA.ANNOTATION_DIR = b'data/ava/annotations'

__C.AVA.FPS = 30

# Since evaluation takes time, during training we can optionally use only a
# subset of AVA validation set to track training progress.
__C.AVA.FULL_EVAL_DURING_TRAINING = False
# During training we use both GT boxes and predicted boxes. This option controls
# the score threshold for the predicted boxes to use.
__C.AVA.DETECTION_SCORE_THRESH_TRAIN = 0.9
# We test on boxes with score >= 0.85 by default.
__C.AVA.DETECTION_SCORE_THRESH_EVAL = [0.85]
# We use boxes with score >= 0.9 to construct LFB.
__C.AVA.LFB_DETECTION_SCORE_THRESH = 0.9

__C.AVA.TRAIN_ON_TRAIN_VAL = False
__C.AVA.TEST_ON_TEST_SET = False

__C.AVA.TRAIN_LISTS = [b'train.csv']
__C.AVA.TEST_LISTS = [b'val.csv']

__C.AVA.TRAIN_BOX_LISTS = [b'ava_train_v2.1.csv', b'ava_train_predicted_boxes.csv']
__C.AVA.TEST_BOX_LISTS = [b'ava_val_predicted_boxes.csv']

__C.AVA.TRAIN_LFB_BOX_LISTS = [b'ava_train_predicted_boxes.csv']
__C.AVA.TEST_LFB_BOX_LISTS = [b'ava_val_predicted_boxes.csv']

__C.AVA.TEST_MULTI_CROP = False
__C.AVA.TEST_MULTI_CROP_SCALES = [224, 256, 320]
__C.AVA.FORCE_TEST_FLIP = False

# Max number of features per time step in LFB.
# We enforce this in order to make an LFB fix-sized to simplify implementation.
__C.AVA.LFB_MAX_NUM_FEAT_PER_STEP = 5

__C.EPIC = AttrDict()
__C.EPIC.FRAME_LIST_DIR = b'data/epic/frame_lists'
__C.EPIC.ANNOTATION_DIR = b'data/epic/annotations'

__C.EPIC.TRAIN_LISTS = [b'train.csv']
__C.EPIC.TEST_LISTS = [b'val.csv']

__C.EPIC.ANNOTATIONS = b'EPIC_train_action_labels.csv'
__C.EPIC.FPS = 30

__C.EPIC.CLASS_TYPE = b''

# This defines how densely we sample clips to construct LFB for a Verb model.
# (in number of clips per second.)
__C.EPIC.VERB_LFB_CLIPS_PER_SECOND = 1
# This defines how densely we sample frames to construct LFB for a Noun model.
# (in number of frames per second.)
__C.EPIC.NOUN_LFB_FRAMES_PER_SECOND = 1
# Max number of features per frame to construct "Noun LFB".
# We enforce this in order to make an LFB fix-sized to simplify implementation.
__C.EPIC.MAX_NUM_FEATS_PER_NOUN_LFB_FRAME = 10

__C.CHARADES = AttrDict()
__C.CHARADES.FRAME_LIST_DIR = b'data/charades/frame_lists'
__C.CHARADES.TRAIN_LISTS = [b'train.csv']
__C.CHARADES.TEST_LISTS = [b'val.csv']

__C.CHARADES.FPS = 24

# To save time, during training we evaluate only 9 crops by default,
# i.e., (left, center, right) * 3 clips.
__C.CHARADES.NUM_TEST_CLIPS_DURING_TRAINING = 9
# Our final test results are combined from 3 spatial shifts
# (left, center, right) * 10 clips.
__C.CHARADES.NUM_TEST_CLIPS_FINAL_EVAL = 30
# This defines how densely we sample clips to construct LFB.
# (in number of clips per second.)
__C.CHARADES.LFB_CLIPS_PER_SECOND = 2

__C.ROI = AttrDict()
# Our default network downsamples spatial dimension by 16x.
__C.ROI.SCALE_FACTOR = 16
__C.ROI.XFORM_RESOLUTION = 7

__C.LFB = AttrDict()
__C.LFB.ENABLED = False

# This is the model param file we'll use to infer features and consturct LFB if
# LOAD_LFB is False.
__C.LFB.MODEL_PARAMS_FILE = b''
# We can optionally store the constructed LFB into a pickle file and reuse it
# next time.
__C.LFB.WRITE_LFB = False
# If true, will load a sotred LFB rather than inferring and constructing a new
# one from LFB.MODEL_PARAMS_FILE.
__C.LFB.LOAD_LFB = False
# The path to the foler containing the train/val LFBs to load.
__C.LFB.LOAD_LFB_PATH = b''

__C.LFB.LFB_DIM = 2048

# For AVA, it's the number of time steps.
# For EPIC-Kitchens and Charades, it's number of featurs.
__C.LFB.WINDOW_SIZE = 100

# General FBO design.
__C.LFB.FBO_TYPE = b'nl'

# FBO-NL design.
__C.FBO_NL = AttrDict()
# Variants
__C.FBO_NL.NUM_LAYERS = 2
__C.FBO_NL.PRE_ACT = True
__C.FBO_NL.PRE_ACT_LN = True
__C.FBO_NL.SCALE = True
__C.FBO_NL.LATENT_DIM = 512

__C.FBO_NL.INPUT_REDUCE_DIM = True
__C.FBO_NL.DROPOUT_RATE = 0.2
__C.FBO_NL.INPUT_DROPOUT_ON = True
__C.FBO_NL.LFB_DROPOUT_ON = True
__C.FBO_NL.NL_DROPOUT_ON = True

__C.IMG_LOAD_RETRY = 10
# Just used as a global variable. No need to manually set it.
__C.GET_TRAIN_LFB = False


def print_cfg():
    import pprint
    logger.info('Config:')
    logger.info(pprint.pformat(__C))


def assert_and_infer_cfg():

    if __C.SOLVER.STEPS is None:
        # Example input: [150150, 150150, 150150]
        __C.SOLVER.STEPS = []
        __C.SOLVER.STEPS.append(0)
        for idx in range(len(__C.SOLVER.STEP_SIZES)):
            __C.SOLVER.STEPS.append(
                __C.SOLVER.STEP_SIZES[idx] + __C.SOLVER.STEPS[idx])
        # Example output: [0, 150150, 300300, 450450]

    assert __C.TRAIN.BATCH_SIZE % __C.NUM_GPUS == 0, \
        "Train batch size should be multiple of num_gpus."

    assert __C.TEST.BATCH_SIZE % __C.NUM_GPUS == 0, \
        "Test batch size should be multiple of num_gpus."

    # Only for AVA.
    __C.LFB.NUM_LFB_FEAT = __C.AVA.LFB_MAX_NUM_FEAT_PER_STEP * __C.LFB.WINDOW_SIZE


def merge_dicts(dict_a, dict_b):
    from ast import literal_eval
    for key, value in dict_a.items():
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # The types must match, too.
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
                raise ValueError(
                    'Type mismatch ({} vs. {}) for config key: {}'.format(
                        type(dict_b[key]), type(value), key)
                )
        # Recursively merge dicts.
        if isinstance(value, AttrDict):
            try:
                merge_dicts(dict_a[key], dict_b[key])
            except BaseException:
                raise Exception('Error under config key: {}'.format(key))
        else:
            dict_b[key] = value


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen))
    merge_dicts(yaml_config, __C)


def cfg_from_list(args_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(args_list) % 2 == 0, 'Specify values or keys for args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # Handle the case when v is a string literal.
            val = literal_eval(value)
        except BaseException:
            val = value
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(val), type(cfg[subkey]))
        cfg[subkey] = val
