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

"""3D ResNet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from core.config import config as cfg
from utils.misc import get_batch_size
import models.head_helper as head_helper
import models.resnet_helper as resnet_helper

logger = logging.getLogger(__name__)


BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}


def obtain_arc(arc_type):
    """
    Architecture definition.
    This function defines the temporal kernel radius and temporal strides for
    each layer in a ResNet.
    For example, use_temp_convs = 1 stands for a temporal kernel size of 3.
    In ResNet50, it has (3, 4, 6, 3) blocks in conv2, 3, 4, 5.
    so the lengths of the corresponding lists are (3, 4, 6, 3).
    """
    pool_stride = 1

    # C2D, ResNet50.
    if arc_type == 1:
        use_temp_convs_1 = [0]
        temp_strides_1 = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 6
        temp_strides_4 = [1, ] * 6
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # I3D, ResNet50.
    if arc_type == 2:
        use_temp_convs_1 = [2]
        temp_strides_1 = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # C2D, ResNet101.
    if arc_type == 3:
        use_temp_convs_1 = [0]
        temp_strides_1 = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 23
        temp_strides_4 = [1, ] * 23
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # I3D, ResNet101.
    if arc_type == 4:
        use_temp_convs_1 = [2]
        temp_strides_1 = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = []
        for i in range(23):
            if i % 2 == 0:
                use_temp_convs_4.append(1)
            else:
                use_temp_convs_4.append(0)

        temp_strides_4 = [1, ] * 23
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5 = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    use_temp_convs_set = [
        use_temp_convs_1,
        use_temp_convs_2,
        use_temp_convs_3,
        use_temp_convs_4,
        use_temp_convs_5
    ]
    temp_strides_set = [
        temp_strides_1,
        temp_strides_2,
        temp_strides_3,
        temp_strides_4,
        temp_strides_5
    ]
    return use_temp_convs_set, temp_strides_set, pool_stride


def create_model(model, data, labels, split, lfb_infer_only, suffix=''):
    """This function defines the full model."""

    cfg.DILATIONS = 1

    group = cfg.RESNETS.NUM_GROUPS
    width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
    batch_size = get_batch_size(split)

    logger.info(
        '--------------- ResNet-{} {}x{}d-{}, {}, {}, infer LFB? {}, suffix: \"{}\" ---------------'.format(
            cfg.MODEL.DEPTH,
            group, width_per_group,
            cfg.RESNETS.TRANS_FUNC,
            cfg.DATASET,
            split,
            lfb_infer_only,
            suffix))

    assert cfg.MODEL.DEPTH in BLOCK_CONFIG.keys(), \
        'Block config is not defined for specified model depth.'
    (n1, n2, n3, n4) = BLOCK_CONFIG[cfg.MODEL.DEPTH]

    res_block = resnet_helper._generic_residual_block_3d
    dim_inner = group * width_per_group

    crop_size = (cfg.TRAIN.CROP_SIZE if (split == 'train' and not lfb_infer_only)
                 else cfg.TEST.CROP_SIZE)

    out_spatial_dim = crop_size // 16

    use_temp_convs_set, temp_strides_set, pool_stride = \
        obtain_arc(cfg.MODEL.VIDEO_ARC_CHOICE)
    logger.info("use_temp_convs_set: {}".format(use_temp_convs_set))
    logger.info("temp_strides_set: {}".format(temp_strides_set))

    conv_blob = model.ConvNd(
        data,
        'conv1',
        3,
        64,
        [1 + use_temp_convs_set[0][0] * 2, 7, 7],
        strides=[temp_strides_set[0][0], 2, 2],
        pads=[use_temp_convs_set[0][0], 3, 3] * 2,
        weight_init=('MSRAFill', {}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=1
    )

    test_mode = False if split not in ['test', 'val'] else True
    if cfg.MODEL.USE_AFFINE:
        bn_blob = model.AffineNd(conv_blob, 'res_conv1_bn', 64)
    else:
        bn_blob = model.SpatialBN(
            conv_blob, 'res_conv1_bn', 64, epsilon=cfg.MODEL.BN_EPSILON,
            momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
        )
    relu_blob = model.Relu(bn_blob, bn_blob)
    max_pool = model.MaxPool(
        relu_blob,
        'pool1',
        kernels=[1, 3, 3],
        strides=[1, 2, 2],
        pads=[0, 1, 1] * 2
    )

    if cfg.MODEL.DEPTH in [50, 101]:
        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            model,
            res_block,
            max_pool,
            64,
            256,
            stride=1,
            num_blocks=n1,
            prefix='res2',
            dim_inner=dim_inner, group=group,
            use_temp_convs=use_temp_convs_set[1],
            temp_strides=temp_strides_set[1]
        )

        layer_mod = cfg.NONLOCAL.LAYER_MOD
        if cfg.MODEL.DEPTH == 101:
            layer_mod = 2
        if not cfg.NONLOCAL.CONV3_NONLOCAL:
            layer_mod = 1000

        blob_in = model.MaxPool(
            blob_in,
            'pool2',
            kernels=[2, 1, 1],
            strides=[2, 1, 1],
            pads=[0, 0, 0] * 2
        )

        if not cfg.MODEL.USE_AFFINE:
            blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                model,
                res_block,
                blob_in,
                dim_in,
                512,
                stride=2,
                num_blocks=n2,
                prefix='res3',
                dim_inner=dim_inner * 2,
                group=group,
                use_temp_convs=use_temp_convs_set[2],
                temp_strides=temp_strides_set[2],
                batch_size=batch_size,
                nonlocal_name='nonlocal_conv3',
                nonlocal_mod=layer_mod
            )
        else:
            blob_in, dim_in = resnet_helper.res_stage_nonlocal_group(
                model,
                res_block,
                blob_in,
                dim_in,
                512,
                stride=2,
                num_blocks=n2,
                prefix='res3',
                dim_inner=dim_inner * 2,
                group=group,
                use_temp_convs=use_temp_convs_set[2],
                temp_strides=temp_strides_set[2],
                batch_size=batch_size,
                pool_stride=pool_stride,
                spatial_dim=(int(crop_size / 8)),
                group_size=4,
                nonlocal_name='nonlocal_conv3',
                nonlocal_mod=layer_mod
            )

        layer_mod = cfg.NONLOCAL.LAYER_MOD
        if cfg.MODEL.DEPTH == 101:
            layer_mod = layer_mod * 4 - 1
        if not cfg.NONLOCAL.CONV4_NONLOCAL:
            layer_mod = 1000

        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            model,
            res_block,
            blob_in,
            dim_in,
            1024,
            stride=2,
            num_blocks=n3,
            prefix='res4',
            dim_inner=dim_inner * 4,
            group=group,
            use_temp_convs=use_temp_convs_set[3],
            temp_strides=temp_strides_set[3],
            batch_size=batch_size,
            nonlocal_name='nonlocal_conv4',
            nonlocal_mod=layer_mod
        )
        if cfg.MODEL.DILATIONS_AFTER_CONV5:
            cfg.DILATIONS = 2

        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            model, res_block, blob_in, dim_in, 2048, stride=1,
            num_blocks=n4,
            prefix='res5', dim_inner=dim_inner * 8, group=group,
            use_temp_convs=use_temp_convs_set[4],
            temp_strides=temp_strides_set[4])

    else:
        raise Exception("Unsupported network settings.")

    if cfg.MODEL.FREEZE_BACKBONE:
        model.StopGradient(blob_in, blob_in)

    if cfg.DATASET == 'ava':
        head_func = head_helper.add_roi_head
    elif cfg.DATASET in ['charades', 'epic']:
        head_func = head_helper.add_basic_head
    else:
        raise NotImplementedError(
            'Unknown dataset {}'.format(cfg.DATASET))

    blob_out, dim_in = head_func(
        model, blob_in, dim_in, pool_stride, out_spatial_dim,
        suffix, lfb_infer_only, test_mode
    )

    if lfb_infer_only:
        return model, None, None

    if cfg.TRAIN.DROPOUT_RATE > 0 and not test_mode:
        blob_out = model.Dropout(
            blob_out, blob_out + '_dropout',
            ratio=cfg.TRAIN.DROPOUT_RATE, is_test=False)

    blob_out = model.FC(
        blob_out, 'pred', dim_in, cfg.MODEL.NUM_CLASSES,
        weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.})
    )

    scale = 1. / cfg.NUM_GPUS
    if split == 'train':
        if cfg.MODEL.MULTI_LABEL:
            prob = model.Sigmoid(blob_out, 'prob')
            loss = model.SigmoidCrossEntropyLoss(
                [blob_out, labels], ['loss'], scale=scale)
        else:
            prob, loss = model.SoftmaxWithLoss(
                [blob_out, labels], ['prob', 'loss'], scale=scale)
    else:
        if cfg.MODEL.MULTI_LABEL:
            prob = model.Sigmoid(
                blob_out, 'prob', engine='CUDNN')
        else:
            prob = model.Softmax(blob_out, 'prob')

        loss = None

    return model, prob, loss
