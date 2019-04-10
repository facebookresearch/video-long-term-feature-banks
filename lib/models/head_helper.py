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

"""Output heads of a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from core.config import config as cfg
import models.lfb_helper as lfb_helper


logger = logging.getLogger(__name__)


def add_basic_head(model, blob_in, dim_in, pool_stride, out_spatial_dim,
             suffix, lfb_infer_only, test_mode):
    """Add an output head for models predicting "clip-level outputs"."""

    # -> (B, 2048, 1, 1, 1)
    pooled = model.AveragePool(
        blob_in, blob_in + '_pooled',
        kernels=[pool_stride, out_spatial_dim, out_spatial_dim],
        strides=[1, 1, 1], pads=[0, 0, 0] * 2)

    all_heads = [pooled]
    new_dim_in = [dim_in]

    if cfg.LFB.ENABLED and not lfb_infer_only:

        fbo_out, fbo_out_dim = lfb_helper.add_fbo_head(
            model, pooled, dim_in,
            num_lfb_feat=cfg.LFB.WINDOW_SIZE,
            test_mode=test_mode, suffix=suffix)

        all_heads.append(fbo_out)
        new_dim_in.append(fbo_out_dim)

    return (model.net.Concat(all_heads,
                             ['pool5', 'pool5_concat_info'],
                             axis=1)[0],
            sum(new_dim_in))


def add_roi_head(model, blob_in, dim_in, pool_stride, out_spatial_dim,
             suffix, lfb_infer_only, test_mode):
    """Add an output head for models predicting "box-level outputs"."""

    # (B, 2048, 16, 14, 14) -> (N, 2048, 1, 1, 1)
    roi_feat = roi_pool(model, blob_in, dim_in, out_spatial_dim, suffix)

    all_heads = [roi_feat]
    new_dim_in = [dim_in]

    if cfg.LFB.ENABLED and not lfb_infer_only:

        fbo_out, fbo_out_dim = lfb_helper.add_fbo_head(
            model, roi_feat, dim_in,
            num_lfb_feat=(cfg.LFB.WINDOW_SIZE
                          * cfg.AVA.LFB_MAX_NUM_FEAT_PER_STEP),
            test_mode=test_mode, suffix=suffix)

        all_heads.append(fbo_out)
        new_dim_in.append(fbo_out_dim)

    return (model.net.Concat(all_heads,
                             ['pool5', 'pool5_concat_info'],
                             axis=1)[0],
            sum(new_dim_in))


def roi_pool(model, blob_in, dim_in, out_spatial_dim, suffix):
    """RoI pooling."""

    # (B, C, 16, 14, 14) -> (B, C, 1, 14, 14)
    blob_pooled = model.AveragePool(
        blob_in,
        'blob_pooled',
        kernels=[cfg.TRAIN.VIDEO_LENGTH // 2, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2
    )

    # (B, C, 1, 14, 14), (B, C, 14, 14)
    blob_pooled = model.Squeeze(blob_pooled, blob_pooled + '_4d', dims=[2])

    # (B, C, 14, 14) and (N, 5) -> (N, C, 7, 7)
    resolution = cfg.ROI.XFORM_RESOLUTION
    roi_feat = lfb_helper.RoIFeatureTransform(
        model, blob_pooled, 'roi_feat_3d',
        spatial_scale=(1.0 / cfg.ROI.SCALE_FACTOR),
        resolution=resolution,
        blob_rois='proposals{}'.format(suffix))

    if resolution > 1:
        # -> (N, C, 1, 1)
        roi_feat = model.MaxPool(
            roi_feat, 'roi_feat_1d', kernels=[resolution, resolution],
            strides=[1, 1], pads=[0, 0] * 2)

    # (N, C, 1, 1) -> (N, C, 1, 1, 1)
    roi_feat, _ = model.Reshape(
        roi_feat,
        ['box_pooled', 'roi_feat_re2_shape'],
        shape=(-1, dim_in, 1, 1, 1))

    return roi_feat
