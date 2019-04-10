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

"""
This script contains functions to build training/testing model abtractions
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np

from caffe2.proto import caffe2_pb2
from caffe2.python import cnn
from caffe2.python import core
from caffe2.python import data_parallel_model
from caffe2.python import dyndep
from caffe2.python import scope
from caffe2.python import workspace

from core.config import config as cfg
from datasets.dataloader import DataLoader
from datasets.dataloader import get_input_db
from models import resnet_video
import utils.lr_policy as lr_policy
import utils.misc as misc

logger = logging.getLogger(__name__)


model_creator_map = {
    'resnet_video': resnet_video,
}


class ModelBuilder(cnn.CNNModelHelper):

    def __init__(self, **kwargs):
        kwargs['order'] = 'NCHW'
        self.train = kwargs.get('train', False)
        self.split = kwargs.get('split', 'train')
        self.force_fw_only = kwargs.get('force_fw_only', False)

        if 'train' in kwargs:
            del kwargs['train']
        if 'split' in kwargs:
            del kwargs['split']
        if 'force_fw_only' in kwargs:
            del kwargs['force_fw_only']

        super(ModelBuilder, self).__init__(**kwargs)
        # Keeping this here in case we have some other params in future to try
        # This is not used for biases anymore
        self.do_not_update_params = []
        self.data_loader = None
        self.input_db = None

        self.current_lr = 0
        self.SetCurrentLr(0)

    def TrainableParams(self, scope=''):
        return [param for param in self.params
                if (param in self.param_to_grad
                    and param not in self.do_not_update_params
                    and (scope == '' or str(param).find(scope) == 0))]

    def build_model(self, suffix, lfb=None,
                    lfb_infer_only=False, shift=1, node_id=0):
        self.input_db = get_input_db(
            dataset=cfg.DATASET, data_type=self.split,
            model=self,
            suffix=suffix,
            lfb=lfb,
            lfb_infer_only=lfb_infer_only,
            shift=shift,
        )

        self.crop_size = misc.get_crop_size(self.split)
        batch_size = misc.get_batch_size(self.split)

        self.data_loader = DataLoader(
            split=self.split, input_db=self.input_db,
            batch_size=batch_size,
            minibatch_queue_size=cfg.MINIBATCH_QUEUE_SIZE,
            crop_size=self.crop_size,
            suffix=suffix,
        )
        self.create_data_parallel_model(
            model=self, db_loader=self.data_loader,
            split=self.split, node_id=node_id,
            train=self.train, force_fw_only=self.force_fw_only,
            suffix=suffix,
            lfb_infer_only=lfb_infer_only,
        )

    def create_data_parallel_model(
        self, model, db_loader, split, node_id,
        train=True, force_fw_only=False, suffix='', lfb_infer_only=False,
    ):
        forward_pass_builder_fun = create_model(
            model=self, split=split, suffix=suffix, lfb_infer_only=lfb_infer_only,
        )

        input_builder_fun = add_inputs(
            model=model, data_loader=db_loader, suffix=suffix
        )

        if train and not force_fw_only:
            param_update_builder_fun = add_parameter_update_ops(model=model)
        else:
            param_update_builder_fun = None
        first_gpu = cfg.ROOT_GPU_ID
        gpus = range(first_gpu, first_gpu + cfg.NUM_GPUS)

        rendezvous_ctx = None

        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=input_builder_fun,
            forward_pass_builder_fun=forward_pass_builder_fun,
            param_update_builder_fun=param_update_builder_fun,
            devices=gpus,
            rendezvous=rendezvous_ctx,
            broadcast_computed_params=False,
            optimize_gradient_memory=cfg.MODEL.MEMONGER,
            use_nccl=not cfg.DEBUG,
        )

    def start_data_loader(self):
        logger.info("Starting data loader...")
        self.data_loader.register_sigint_handler()
        self.data_loader.start()
        self.data_loader.prefill_minibatch_queue()

    def shutdown_data_loader(self):
        logger.info("Shuting down data loader...")
        self.data_loader.shutdown_dataloader()

    def Relu_(self, blob_in):
        """ReLU with inplace option."""
        blob_out = self.Relu(
            blob_in,
            blob_in if cfg.MODEL.ALLOW_INPLACE_RELU else blob_in + "_relu")
        return blob_out

    def Conv3dBN(
        self, blob_in, prefix, dim_in, dim_out, kernels, strides, pads,
        group=1, bn_init=None,
        **kwargs
    ):
        conv_blob = self.ConvNd(
            blob_in, prefix, dim_in, dim_out, kernels, strides=strides,
            pads=pads, group=group,
            weight_init=("MSRAFill", {}),
            bias_init=('ConstantFill', {'value': 0.}), no_bias=1)
        blob_out = self.SpatialBN(
            conv_blob, prefix + "_bn", dim_out,
            epsilon=cfg.MODEL.BN_EPSILON,
            momentum=cfg.MODEL.BN_MOMENTUM,
            is_test=self.split in ['test', 'val'])

        if bn_init is not None and bn_init != 1.0:
            self.param_init_net.ConstantFill(
                [prefix + "_bn_s"],
                prefix + "_bn_s", value=bn_init)

        return blob_out

    # Conv + Affine wrapper.
    def Conv3dAffine(
        self, blob_in, prefix, dim_in, dim_out, kernels, strides, pads,
        group=1,
        suffix='_bn',
        inplace_affine=False,
        dilations=None,
        **kwargs
    ):

        if dilations is None:
            dilations = [1, 1, 1]
        conv_blob = self.ConvNd(
            blob_in, prefix, dim_in, dim_out, kernels, strides=strides,
            pads=pads, group=group,
            weight_init=("MSRAFill", {}),
            bias_init=('ConstantFill', {'value': 0.}),
            no_bias=1,
            dilations=dilations)
        blob_out = self.AffineNd(
            conv_blob, prefix + suffix, dim_out, inplace=inplace_affine)

        return blob_out

    def AffineNd(
            self, blob_in, blob_out, dim_in, share_with=None, inplace=False):
        blob_out = blob_out or self.net.NextName()
        is_not_sharing = share_with is None
        param_prefix = blob_out if is_not_sharing else share_with
        weight_init = ('ConstantFill', {'value': 1.})
        bias_init = ('ConstantFill', {'value': 0.})
        scale = self.param_init_net.__getattr__(weight_init[0])(
            [],
            param_prefix + '_s',
            shape=[dim_in, ],
            **weight_init[1]
        )
        bias = self.param_init_net.__getattr__(bias_init[0])(
            [],
            param_prefix + '_b',
            shape=[dim_in, ],
            **bias_init[1]
        )
        if is_not_sharing:
            self.net.Proto().external_input.extend([str(scale), str(bias)])
            self.params.extend([scale, bias])
            self.weights.append(scale)
            self.biases.append(bias)
        if inplace:
            return self.net.AffineNd([blob_in, scale, bias], blob_in)
        else:
            return self.net.AffineNd([blob_in, scale, bias], blob_out)

    def SetCurrentLr(self, cur_iter):
        """Set the model's current learning rate without changing any blobs in
        the workspace.
        """
        self.current_lr = lr_policy.get_lr_at_iter(cur_iter)

    def UpdateWorkspaceLr(self, cur_iter):
        """Updates the model's current learning rate and the workspace (learning
        rate and update history/momentum blobs).
        """
        new_lr = lr_policy.get_lr_at_iter(cur_iter)
        if new_lr != self.current_lr:
            ratio = _get_lr_change_ratio(self.current_lr, new_lr)
            if ratio > 1.1:
                logger.info(
                    'Setting learning rate to {:.6f} at iteration {}'.format(
                        new_lr, cur_iter))
            self._SetNewLr(self.current_lr, new_lr)

    def _SetNewLr(self, cur_lr, new_lr):
        """Do the actual work of updating the model and workspace blobs.
        """
        assert cur_lr > 0
        for i in range(cfg.NUM_GPUS):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
                workspace.FeedBlob(
                    'gpu_{}/lr'.format(i), np.array(new_lr, dtype=np.float32))

        ratio = _get_lr_change_ratio(cur_lr, new_lr)
        if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            self._CorrectMomentum(new_lr / cur_lr)
        self.current_lr = new_lr

    def _CorrectMomentum(self, correction):
        """The MomentumSGDUpdate op implements the update V as

            V := mu * V + lr * grad,

        where mu is the momentum factor, lr is the learning rate, and grad is the
        stochastic gradient. Since V is not defined independently of the learning
        rate (as it should ideally be), when the learning rate is changed we should
        scale the update history V in order to make it compatible in scale with
        lr * grad.
        """
        # Avoid noisy logging.
        if correction < 0.9 or correction > 1.1:
            logger.info('Scaling update history by {:.6f} (new/old lr)'.format(
                correction))

        root_gpu_id = cfg.ROOT_GPU_ID
        num_gpus = cfg.NUM_GPUS
        for i in range(root_gpu_id, root_gpu_id + num_gpus):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
                with core.NameScope("gpu_{}".format(i)):
                    params = self.GetParams()
                    for param in params:
                        if param in self.TrainableParams():
                            op = core.CreateOperator(
                                'Scale', [param + '_momentum'],
                                [param + '_momentum'],
                                scale=correction)
                            workspace.RunOperatorOnce(op)


def create_model(model, split, suffix, lfb_infer_only):
    model_name = cfg.MODEL.MODEL_NAME
    assert model_name in model_creator_map.keys(), \
        'Unknown model_type {}'.format(model_name)

    def model_creator(model, loss_scale):
        model, softmax, loss = model_creator_map[model_name].create_model(
            model=model,
            data="data{}".format(suffix),
            labels="labels{}".format(suffix),
            split=split,
            suffix=suffix, lfb_infer_only=lfb_infer_only,
        )
        return [loss]

    return model_creator


def add_inputs(model, data_loader, suffix):
    blob_names = data_loader.get_blob_names()
    queue_name = data_loader._blobs_queue_name

    def input_fn(model):
        for blob_name in blob_names:
            workspace.CreateBlob(scope.CurrentNameScope() + blob_name)
        model.DequeueBlobs(queue_name, blob_names)
        model.StopGradient('data{}'.format(suffix), 'data{}'.format(suffix))

    return input_fn


def add_parameter_update_ops(model):
    def param_update_ops(model):
        lr = model.param_init_net.ConstantFill(
            [], 'lr', shape=[1], value=model.current_lr)
        weight_decay = model.param_init_net.ConstantFill(
            [], 'weight_decay', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY
        )
        weight_decay_bn = model.param_init_net.ConstantFill(
            [], 'weight_decay_bn', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY_BN
        )
        one = model.param_init_net.ConstantFill(
            [], "ONE", shape=[1], value=1.0
        )
        params = model.GetParams()
        curr_scope = scope.CurrentNameScope()
        # scope is of format 'gpu_{}/'.format(gpu_id), so remove the separator.
        trainable_params = model.TrainableParams(curr_scope[:-1])
        assert len(params) > 0, 'No trainable params found in model'
        for param in params:
            # Only update trainable params.
            if param in trainable_params:
                param_grad = model.param_to_grad[param]
                # The param grad is the summed gradient for the parameter across
                # all gpus/hosts.
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0)

                if '_bn' in str(param):
                    model.WeightedSum(
                        [param_grad, one, param, weight_decay_bn],
                        param_grad)
                else:
                    model.WeightedSum(
                        [param_grad, one, param, weight_decay],
                        param_grad)
                model.net.MomentumSGDUpdate(
                    [param_grad, param_momentum, lr, param],
                    [param_grad, param_momentum, param],
                    momentum=cfg.SOLVER.MOMENTUM,
                    nesterov=cfg.SOLVER.NESTEROV,
                )
    return param_update_ops


def _get_lr_change_ratio(cur_lr, new_lr):
    eps = 1e-10
    ratio = np.max(
        (new_lr / np.max((cur_lr, eps)), cur_lr / np.max((new_lr, eps)))
    )
    return ratio
