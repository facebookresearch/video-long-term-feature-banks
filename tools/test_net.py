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

"""Test a video model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import workspace
import argparse
import logging
import numpy as np
import os
import sys

from core.config import assert_and_infer_cfg
from core.config import cfg_from_file
from core.config import cfg_from_list
from core.config import config as cfg
from core.config import print_cfg
from lfb_loader import get_lfb
from models import model_builder_video
from utils.timer import Timer
import utils.c2 as c2_utils
import utils.checkpoints as checkpoints
import utils.metrics as metrics
import utils.misc as misc

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def test_net(lfb=None):
    """
    Test a model.
    For AVA, we can either test on a center crop or multiple crops.
    For EPIC-Kitchens, we test on a center crop for simplicity.
    For Charades, we follow prior work (e.g. non-local net) and perform
    3-spatial-shifts * 10-clip testing.
    """

    if cfg.DATASET == 'ava':
        for threshold in cfg.AVA.DETECTION_SCORE_THRESH_EVAL:
            cfg.AVA.DETECTION_SCORE_THRESH = threshold

            if cfg.AVA.TEST_MULTI_CROP:
                cfg.LFB.WRITE_LFB = False
                cfg.LFB.LOAD_LFB = False

                for flip in [False, True]:
                    cfg.AVA.FORCE_TEST_FLIP = flip

                    for scale in cfg.AVA.TEST_MULTI_CROP_SCALES:
                        cfg.TEST.SCALE = scale
                        cfg.TEST.CROP_SIZE = min(256, scale)

                        lfb = None
                        for shift in range(3):
                            out_name = 'detections_%s.csv' % \
                                get_test_name(shift)
                            if os.path.isfile(out_name):
                                logger.info("%s already exists." % out_name)
                                continue

                            if cfg.LFB.ENABLED and lfb is None:
                                lfb = get_lfb(cfg.LFB.MODEL_PARAMS_FILE,
                                              is_train=False)

                            test_one_crop(lfb=lfb,
                                          suffix='_final_test',
                                          shift=shift)
                metrics.combine_ava_multi_crops()
            else:
                test_one_crop(lfb=lfb, suffix='_final_test')
    else:
        if cfg.DATASET == 'charades':
            cfg.CHARADES.NUM_TEST_CLIPS = cfg.CHARADES.NUM_TEST_CLIPS_FINAL_EVAL
        test_one_crop(lfb=lfb, suffix='_final_test')


def test_one_crop(lfb=None, suffix='', shift=None):
    """Test one crop."""
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)

    cfg.AVA.FULL_EVAL = True

    if lfb is None and cfg.LFB.ENABLED:
        print_cfg()
        lfb = get_lfb(cfg.LFB.MODEL_PARAMS_FILE, is_train=False)

    print_cfg()

    workspace.ResetWorkspace()
    logger.info("Done ResetWorkspace...")

    timer = Timer()

    logger.warning('Testing started...')  # for monitoring cluster jobs

    if shift is None:
        shift = cfg.TEST.CROP_SHIFT
    test_model = model_builder_video.ModelBuilder(
        train=False,
        use_cudnn=True,
        cudnn_exhaustive_search=True,
        split=cfg.TEST.DATA_TYPE
    )

    test_model.build_model(lfb=lfb, suffix=suffix, shift=shift)

    if cfg.PROF_DAG:
        test_model.net.Proto().type = 'prof_dag'
    else:
        test_model.net.Proto().type = 'dag'

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    misc.save_net_proto(test_model.net)
    misc.save_net_proto(test_model.param_init_net)

    total_test_net_iters = misc.get_total_test_iters(test_model)

    test_model.start_data_loader()
    test_meter = metrics.MetricsCalculator(
        model=test_model,
        split=cfg.TEST.DATA_TYPE,
        video_idx_to_name=test_model.input_db._video_idx_to_name,
        total_num_boxes=(test_model.input_db._num_boxes_used
                         if cfg.DATASET == 'ava' else None))

    if cfg.TEST.PARAMS_FILE:
        checkpoints.load_model_from_params_file_for_test(
            test_model, cfg.TEST.PARAMS_FILE)
    else:
        raise Exception('No params files specified for testing model.')

    for test_iter in range(total_test_net_iters):
        timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        timer.toc()

        if test_iter == 0:
            misc.print_net(test_model)
            os.system('nvidia-smi')

        test_meter.calculate_and_log_all_metrics_test(
            test_iter, timer, total_test_net_iters, suffix)

    test_meter.finalize_metrics(name=get_test_name(shift))
    test_meter.log_final_metrics(test_iter, total_test_net_iters)
    test_model.shutdown_data_loader()


def get_test_name(shift):
    if cfg.DATASET != 'ava':
        return 'final'

    return 'final_%d%s_shift%d_%.03f' % (
        cfg.TEST.SCALE,
        '_flip' if cfg.AVA.FORCE_TEST_FLIP else '',
        shift,
        cfg.AVA.DETECTION_SCORE_THRESH)


def main():
    c2_utils.import_detectron_ops()
    parser = argparse.ArgumentParser(description='Classification model testing')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see configs.py for all options',
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
    test_net()


if __name__ == '__main__':
    main()
