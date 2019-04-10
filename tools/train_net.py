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

"""Train a video model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace
import argparse
import logging
import os
import sys

from core.config import assert_and_infer_cfg
from core.config import cfg_from_file
from core.config import cfg_from_list
from core.config import config as cfg
from core.config import print_cfg
from lfb_loader import get_lfb
from models import model_builder_video
from test_net import test_net
from utils.timer import Timer
import utils.bn_helper as bn_helper
import utils.c2 as c2_utils
import utils.checkpoints as checkpoints
import utils.metrics as metrics
import utils.misc as misc

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def create_wrapper(is_train, lfb=None):
    """
    a simpler wrapper that creates the elements for train/test models
    """
    if is_train:
        suffix = '_train'
        split = cfg.TRAIN.DATA_TYPE
    else:
        suffix = '_test'
        split = cfg.TEST.DATA_TYPE

    model = model_builder_video.ModelBuilder(
        train=is_train,
        use_cudnn=True,
        cudnn_exhaustive_search=True,
        ws_nbytes_limit=(cfg.CUDNN_WORKSPACE_LIMIT * 1024 * 1024),
        split=split,
    )
    model.build_model(suffix=suffix, lfb=lfb)

    if cfg.PROF_DAG:
        model.net.Proto().type = 'prof_dag'
    else:
        model.net.Proto().type = 'dag'

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    model.start_data_loader()

    timer = Timer()
    meter = metrics.MetricsCalculator(
        model=model, split=split,
        video_idx_to_name=model.input_db._video_idx_to_name,
        total_num_boxes=(model.input_db._num_boxes_used if cfg.DATASET == 'ava'
                         else None)
    )

    misc.save_net_proto(model.net)
    misc.save_net_proto(model.param_init_net)

    return model, timer, meter


def train(opts):
    """Train a model."""

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logging.getLogger(__name__)

    # Generate seed.
    misc.generate_random_seed(opts)

    # Create checkpoint dir.
    checkpoint_dir = checkpoints.create_and_get_checkpoint_directory()
    logger.info('Checkpoint directory created: {}'.format(checkpoint_dir))

    # Setting training-time-specific configurations.
    cfg.AVA.FULL_EVAL = cfg.AVA.FULL_EVAL_DURING_TRAINING
    cfg.AVA.DETECTION_SCORE_THRESH = cfg.AVA.DETECTION_SCORE_THRESH_TRAIN
    cfg.CHARADES.NUM_TEST_CLIPS = cfg.CHARADES.NUM_TEST_CLIPS_DURING_TRAINING

    test_lfb, train_lfb = None, None

    if cfg.LFB.ENABLED:
        test_lfb = get_lfb(cfg.LFB.MODEL_PARAMS_FILE, is_train=False)
        train_lfb = get_lfb(cfg.LFB.MODEL_PARAMS_FILE, is_train=True)

    # Build test_model.
    # We build test_model first, so that we don't overwrite init.
    test_model, test_timer, test_meter = create_wrapper(
        is_train=False,
        lfb=test_lfb,
    )
    total_test_iters = misc.get_total_test_iters(test_model)
    logger.info('Test iters: {}'.format(total_test_iters))

    # Build train_model.
    train_model, train_timer, train_meter = create_wrapper(
        is_train=True,
        lfb=train_lfb,
    )

    # Bould BN auxilary model.
    if cfg.TRAIN.COMPUTE_PRECISE_BN:
        bn_aux = bn_helper.BatchNormHelper()
        bn_aux.create_bn_aux_model(node_id=opts.node_id)

    # Load checkpoint or pre-trained weight.
    # See checkpoints.load_model_from_params_file for more details.
    start_model_iter = 0
    if cfg.CHECKPOINT.RESUME or cfg.TRAIN.PARAMS_FILE:
        start_model_iter = checkpoints.load_model_from_params_file(train_model)

    logger.info("------------- Training model... -------------")
    train_meter.reset()
    last_checkpoint = checkpoints.get_checkpoint_resume_file()

    for curr_iter in range(start_model_iter, cfg.SOLVER.MAX_ITER):
        train_model.UpdateWorkspaceLr(curr_iter)

        train_timer.tic()
        # SGD step.
        workspace.RunNet(train_model.net.Proto().name)
        train_timer.toc()

        if curr_iter == start_model_iter:
            misc.print_net(train_model)
            os.system('nvidia-smi')
            misc.show_flops_params(train_model)

        misc.check_nan_losses()

        # Checkpoint.
        if (curr_iter + 1) % cfg.CHECKPOINT.CHECKPOINT_PERIOD == 0 \
                or curr_iter + 1 == cfg.SOLVER.MAX_ITER:
            if cfg.TRAIN.COMPUTE_PRECISE_BN:
                bn_aux.compute_and_update_bn_stats(curr_iter)

            last_checkpoint = os.path.join(
                checkpoint_dir,
                'c2_model_iter{}.pkl'.format(curr_iter + 1))
            checkpoints.save_model_params(
                model=train_model,
                params_file=last_checkpoint,
                model_iter=curr_iter)

        train_meter.calculate_and_log_all_metrics_train(
            curr_iter, train_timer, suffix='_train')

        # Evaluation.
        if (curr_iter + 1) % cfg.TRAIN.EVAL_PERIOD == 0:
            if cfg.TRAIN.COMPUTE_PRECISE_BN:
                bn_aux.compute_and_update_bn_stats(curr_iter)

            test_meter.reset()
            logger.info("=> Testing model")
            for test_iter in range(0, total_test_iters):
                test_timer.tic()
                workspace.RunNet(test_model.net.Proto().name)
                test_timer.toc()

                test_meter.calculate_and_log_all_metrics_test(
                    test_iter, test_timer, total_test_iters, suffix='_test')

            test_meter.finalize_metrics()
            test_meter.compute_and_log_best()
            test_meter.log_final_metrics(curr_iter)

            # Finalize and reset train_meter after test.
            train_meter.finalize_metrics(is_train=True)

            json_stats = metrics.get_json_stats_dict(
                train_meter, test_meter, curr_iter)
            misc.log_json_stats(json_stats)

            train_meter.reset()

    train_model.shutdown_data_loader()
    test_model.shutdown_data_loader()

    if cfg.TRAIN.TEST_AFTER_TRAIN:
        cfg.TEST.PARAMS_FILE = last_checkpoint
        test_net(test_lfb)


def main():
    c2_utils.import_detectron_ops()
    parser = argparse.ArgumentParser(description='Classification model training')
    parser.add_argument('--node_id', type=int, default=0,
                        help='Node id')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)

    assert_and_infer_cfg()
    print_cfg()

    train(args)


if __name__ == '__main__':
    main()
