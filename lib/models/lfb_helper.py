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

"""Helper functions for adding (feature bank operators) FBOs to a model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from core.config import config as cfg

logger = logging.getLogger(__name__)


# init_params1 is used in theta, phi, and g.
init_params1 = {
    'weight_init': ('GaussianFill', {'std': cfg.NONLOCAL.CONV_INIT_STD}),
    'bias_init': ('ConstantFill', {'value': 0.}),
    'no_bias': cfg.NONLOCAL.NO_BIAS}

# init_params2 is used in the output 1x1 conv.
init_params2 = {
    'weight_init': ('ConstantFill', {'value': 0.}),
    'bias_init': ('ConstantFill', {'value': 0.}),
    'no_bias': cfg.NONLOCAL.NO_BIAS}


def NTC_to_NCT11(model, blob_in, dim, num_feat, name=''):
    # (N, N2, 2048) -> (N, 2048, N2)
    blob_in = model.Transpose(
        blob_in, blob_in + '_tr' + name, axes=(0, 2, 1))

    # (N, 2048, N2) -> (N, 2048, N2, 1, 1)
    blob_in, _ = model.Reshape(
        blob_in,
        [blob_in + '_rs' + name, blob_in + '_rs_shape' + name],
        shape=(-1, dim, num_feat, 1, 1))
    return blob_in


def add_fbo_head(model, blob_in, dim_in, num_lfb_feat, test_mode, suffix):
    """Add feature bank operator (FBO)."""
    if cfg.LFB.FBO_TYPE == 'avg':
        return add_fbo_avg_head(
            model, num_lfb_feat, 'fbo_avg_out', suffix)

    elif cfg.LFB.FBO_TYPE == 'max':
        return add_fbo_max_head(
            model, num_lfb_feat, 'fbo_max_out', suffix)

    elif cfg.LFB.FBO_TYPE == 'nl':
        return add_fbo_nl_head(
            model, blob_in,
            dim_in=dim_in,
            num_lfb_feat=num_lfb_feat,
            test_mode=test_mode,
            suffix=suffix,
        )
    else:
        raise NotImplementedError


def add_fbo_nl_head(
        model, blob_in, dim_in, num_lfb_feat, test_mode, suffix):
    """Add "non-local" feature bank operator (FBO-NL)."""

    # -> (N, C, 1, 1, 1)
    blob_in, blob_in_dim = prepare_nl_input(
        model, blob_in, dim_in, '_fbonl', test_mode)

    # -> (N, C, num_lfb_feat, 1, 1)
    lfb = get_lfb_blob(model, num_lfb_feat, suffix)

    # -> (N, C, num_lfb_feat, 1, 1)
    lfb, lfb_dim = prepare_lfb(model, lfb, test_mode, suffix)

    # -> (N, C, 1, 1, 1)
    return (NLLayers(model,
                    A=blob_in,
                    B=lfb,
                    in_dim1=blob_in_dim,
                    in_dim2=lfb_dim,
                    latent_dim=cfg.FBO_NL.LATENT_DIM,
                    num_feat1=1,
                    num_feat2=num_lfb_feat,
                    prefix='lfb',
                    test_mode=test_mode),
            blob_in_dim)


def add_fbo_avg_head(model, num_lfb_feat, out_name, suffix):
    """Add "avg pooling" feature bank operator (FBO-Avg)."""

    # -> (N, lfb_dim, window_size, 1, 1)
    lfb = get_lfb_blob(model, num_lfb_feat, suffix)

    return model.AveragePool(
        lfb, out_name,
        kernels=[num_lfb_feat, 1, 1],
        strides=[1, 1, 1], pads=[0, 0, 0] * 2), cfg.LFB.LFB_DIM


def add_fbo_max_head(model, num_lfb_feat, out_name, suffix):
    """Add "max pooling" feature bank operator (FBO-Avg)."""

    # -> (N, lfb_dim, window_size, 1, 1)
    lfb = get_lfb_blob(model, num_lfb_feat, suffix)

    return model.MaxPool(
        lfb, out_name,
        kernels=[num_lfb_feat, 1, 1],
        strides=[1, 1, 1], pads=[0, 0, 0] * 2), cfg.LFB.LFB_DIM


def RoIFeatureTransform(
    model,
    blobs_in,
    blob_out,
    blob_rois='proposals',
    resolution=7,
    spatial_scale=1. / 16.,
    sampling_ratio=0
):
    """Add the specified RoI pooling method. The sampling_ratio argument
    is supported for some, but not all, RoI transform methods.
    """

    # sampling_ratio is ignored for RoIPoolF.
    xform_out = model.RoIAlign(
        [blobs_in, blob_rois], [blob_out],
        pooled_w=resolution,
        pooled_h=resolution,
        spatial_scale=spatial_scale,
        sampling_ratio=sampling_ratio
    )
    # Only return the first blob (the transformed features).
    return xform_out[0] if isinstance(xform_out, tuple) else xform_out


def get_lfb_blob(model, num_lfb_feat, suffix):
    return NTC_to_NCT11(
        model, 'lfb{}'.format(suffix), cfg.LFB.LFB_DIM, num_lfb_feat)


def pre_act(model, x):
    """Pre-activation style non-linearity."""
    if cfg.FBO_NL.PRE_ACT_LN:
        x = model.LayerNorm(
            x, [x + "_ln",
                x + "_ln_mean",
                x + "_ln_std"])[0]
    return model.Relu(x, x + "_relu")


def NLCore(
        model, in_blob1, in_blob2, in_dim1, in_dim2, latent_dim,
        num_feat1, num_feat2, prefix, test_mode):
    """Core logic of non-local blocks."""

    theta = model.ConvNd(
        in_blob1, prefix + '_theta',
        in_dim1,
        latent_dim,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params1)

    phi = model.ConvNd(
        in_blob2,
        prefix + '_phi',
        in_dim2,
        latent_dim,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params1)

    g = model.ConvNd(
        in_blob2,
        prefix + '_g',
        in_dim2,
        latent_dim,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params1)

    theta, theta_shape_5d = model.Reshape(
        theta,
        [theta + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else theta,
         theta + '_shape5d'],
        shape=(-1, latent_dim, num_feat1))

    phi, phi_shape_5d = model.Reshape(
        phi,
        [phi + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else phi,
         phi + '_shape5d'],
        shape=(-1, latent_dim, num_feat2))

    g, g_shape_5d = model.Reshape(
        g,
        [g + '_re',
         g + '_shape5d'],
        shape=(-1, latent_dim, num_feat2))

    # (N, C, num_feat1), (N, C, num_feat2) -> (N, num_feat1, num_feat2)
    theta_phi = model.net.BatchMatMul(
        [theta, phi], prefix + '_affinity', trans_a=1)

    if cfg.FBO_NL.SCALE:
        theta_phi = model.Scale(
            theta_phi, theta_phi, scale=latent_dim**-.5)

    p = model.Softmax(
        theta_phi, theta_phi + '_prob', engine='CUDNN', axis=2)

    # (N, C, num_feat2), (N, num_feat1, num_feat2) -> (B, C, num_feat1)
    t = model.net.BatchMatMul([g, p], prefix + '_y', trans_b=1)

    blob_out, t_shape = model.Reshape(
        [t, theta_shape_5d],
        [t + '_re' if not cfg.MODEL.ALLOW_INPLACE_RESHAPE else t,
            t + '_shape3d'])

    if cfg.FBO_NL.PRE_ACT:
        blob_out = pre_act(model, blob_out)

    blob_out = model.ConvNd(
        blob_out, prefix + '_out',
        latent_dim,
        in_dim1,
        [1, 1, 1],
        strides=[1, 1, 1],
        pads=[0, 0, 0] * 2,
        **init_params2)

    if not cfg.FBO_NL.PRE_ACT:
        blob_out = model.LayerNorm(
            blob_out,
            [prefix + "_ln", prefix + "_ln_mean", prefix + "_ln_std"])[0]

    if cfg.FBO_NL.LFB_DROPOUT_ON and not test_mode:
        blob_out = model.Dropout(
            blob_out, blob_out + '_drop',
            ratio=cfg.FBO_NL.DROPOUT_RATE, is_test=False)

    return blob_out


def NLLayers(model, A, B, in_dim1, in_dim2, latent_dim,
             num_feat1, num_feat2, prefix, test_mode):
    """Stack NL blocks."""

    for nl_gcn_layer_idx in range(cfg.FBO_NL.NUM_LAYERS):
        prefix_in = prefix + '_nl%d' % nl_gcn_layer_idx

        nl_out = NLCore(
            model,
            in_blob1=A,
            in_blob2=B,
            in_dim1=in_dim1,
            in_dim2=in_dim2,
            latent_dim=latent_dim,
            num_feat1=num_feat1,
            num_feat2=num_feat2,
            prefix=prefix_in,
            test_mode=test_mode,
        )

        nl_out = model.net.Sum(
            [nl_out, A], prefix_in + "_sum")

        if not cfg.FBO_NL.PRE_ACT:
            nl_out = model.Relu(nl_out, prefix_in + "_relu")
        A = nl_out
    return nl_out


def prepare_nl_input(model, blob, dim_in, prefix, test_mode):
    """
    Pre-processing layers (dimensionality reduction and dropout)
    for the input of FBO-NL.
    """

    new_dim = dim_in
    if cfg.FBO_NL.INPUT_REDUCE_DIM:
        blob = model.ConvNd(
            blob, blob + prefix + '_reduc',
            dim_in,
            cfg.FBO_NL.LATENT_DIM,
            [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2,
            weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=cfg.NONLOCAL.NO_BIAS)
        new_dim = cfg.FBO_NL.LATENT_DIM

    if cfg.FBO_NL.INPUT_DROPOUT_ON and not test_mode:
        blob = model.Dropout(
            blob, blob + prefix + '_drop',
            ratio=cfg.FBO_NL.DROPOUT_RATE, is_test=False)
    return blob, new_dim


def prepare_lfb(model, lfb, test_mode, suffix):
    """Pre-processing layers (dimensionality reduction and dropout) for LFB."""

    lfb = model.ConvNd(
        lfb,
        'lfb_1x1',
        cfg.LFB.LFB_DIM,
        cfg.FBO_NL.LATENT_DIM,
        [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2,
        weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
        bias_init=('ConstantFill', {'value': 0.}),
        no_bias=cfg.NONLOCAL.NO_BIAS)

    if cfg.FBO_NL.LFB_DROPOUT_ON and not test_mode:
        lfb = model.Dropout(
            lfb, lfb + '_drop',
            ratio=cfg.FBO_NL.DROPOUT_RATE, is_test=False)

    return lfb, cfg.FBO_NL.LATENT_DIM
