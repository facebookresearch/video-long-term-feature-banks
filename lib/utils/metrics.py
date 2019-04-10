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

"""Evaluation utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cPickle as pickle
import cv2
import datetime
import logging
import numpy as np
import os
import pprint
import sklearn.metrics as metrics
import time

from caffe2.python import workspace
from core.config import config as cfg
import utils.misc as misc

# It's okay if we cannot find ava_eval_helper (for example, when we use
# Charades or EPIC-Kitchens only.)
try:
    from utils.ava_eval_helper import evaluate_ava
    from utils.ava_eval_helper import read_csv
    from utils.ava_eval_helper import read_exclusions
    from utils.ava_eval_helper import read_labelmap
    from utils.ava_eval_helper import evaluate_ava_from_files
except Exception:
    pass

logger = logging.getLogger(__name__)


def get_ava_mini_groundtruth(full_groundtruth):
    """
    Get the groundtruth annotations corresponding the "subset" of AVA val set.
    We define the subset to be the frames such that (second % 4 == 0).
    We optionally use subset for faster evaluation during training
    (in order to track training progress).
    """
    ret = [defaultdict(list), defaultdict(list), defaultdict(list)]

    for i in range(3):
        for key in full_groundtruth[i].keys():
            if int(key.split(',')[1]) % 4 == 0:
                ret[i][key] = full_groundtruth[i][key]
    return ret


class MetricsCalculator():

    def __init__(self, model, split, video_idx_to_name, total_num_boxes):
        self.model = model
        self.split = split
        self.video_idx_to_name = video_idx_to_name
        self._total_num_boxes = total_num_boxes

        self.best_top1 = float('inf')
        self.best_top5 = float('inf')

        self.best_map = float('inf') * (-1.0)
        self.lr = 0  # only used by train.
        self.num_test_clips = 1
        self.reset()

        if cfg.DATASET == "ava":
            # We load AVA annotations only once here, rather than loading
            # them every time we call AVA evaluation code.
            self.excluded_keys = read_exclusions(
                os.path.join(cfg.AVA.ANNOTATION_DIR,
                             "ava_val_excluded_timestamps_v2.1.csv"))

            self.categories, self.class_whitelist = read_labelmap(
                os.path.join(
                    cfg.AVA.ANNOTATION_DIR,
                    "ava_action_list_v2.1_for_activitynet_2018.pbtxt"))

            logger.info("CATEGORIES (%d):\n%s", len(self.categories),
                        pprint.pformat(self.categories, indent=2))

            gt_filename = os.path.join(cfg.AVA.ANNOTATION_DIR, "ava_val_v2.1.csv")
            self.full_groundtruth = read_csv(
                gt_filename,
                self.class_whitelist)

            self.mini_groundtruth = get_ava_mini_groundtruth(
                self.full_groundtruth)

            logger.info('%d (mini: %d) GT boxes loaded from %s.' % (
                len(self.full_groundtruth[0]),
                len(self.mini_groundtruth[0]),
                gt_filename))
            logger.info('%d (mini: %d) GT labels loaded from %s.' % (
                len(self.full_groundtruth[0]),
                len(self.mini_groundtruth[0]),
                gt_filename))
        elif cfg.DATASET == 'charades':
            self.num_test_clips = cfg.CHARADES.NUM_TEST_CLIPS

    def reset(self):
        # This should clear out all the metrics computed so far except for the
        # best_topN metrics.
        logger.info('Resetting {} metrics...'.format(self.split))

        self.aggr_err = 0.0
        self.aggr_err5 = 0.0
        self.all_preds = []
        self.all_labels = []
        self.all_original_boxes = []
        self.all_metadata = []
        self.aggr_loss = 0.0
        self.aggr_batch_size = 0

    def stack_predictions(self):
        """Stack list of predictions and labels into a numpy array."""
        all_preds = np.vstack(self.all_preds)
        all_labels = np.vstack(self.all_labels)
        if not cfg.MODEL.MULTI_LABEL:
            all_labels = all_labels.flatten()
        logger.info(all_preds.shape)
        logger.info(all_labels.shape)
        num_clips_to_use = self.num_test_clips * cfg.TEST.DATASET_SIZE
        num_clips_tested = all_preds.shape[0]
        assert num_clips_tested == all_labels.shape[0]
        assert num_clips_tested >= num_clips_to_use
        assert num_clips_tested - num_clips_to_use < cfg.TEST.BATCH_SIZE

        all_preds = all_preds[:num_clips_to_use]
        all_labels = all_labels[:num_clips_to_use]
        return all_preds, all_labels

    def aggregate_predictions_from_clips(self):
        """
        Charades is a video level task and the standard practice is to sample a
        fixed number of clips uniformaly and aggregate the predictions.
        This function performs the aggregation.
        """
        all_preds, all_labels = self.stack_predictions()
        actual_num_videos = all_preds.shape[0] // self.num_test_clips

        for i in range(actual_num_videos):
            for clip in range(1, self.num_test_clips):

                cur_clip_index = i + clip * actual_num_videos
                assert np.array_equal(all_labels[i],
                                      all_labels[cur_clip_index]), \
                    (i, clip, actual_num_videos,
                     all_labels[i], all_labels[cur_clip_index])
                all_preds[i] = np.maximum(
                    all_preds[cur_clip_index], all_preds[i])

        self.all_labels = all_labels[:actual_num_videos]
        self.all_preds = all_preds[:actual_num_videos]

    def finalize_metrics(self, is_train=False, name='latest'):
        """Finalize testing and compute the final metrics."""

        self.avg_loss = self.aggr_loss / self.aggr_batch_size
        if cfg.MODEL.MULTI_LABEL:
            if is_train:
                self.full_map = 0.0
            else:
                if cfg.DATASET == 'charades':
                    if self.num_test_clips > 1:
                        self.aggregate_predictions_from_clips()
                    self.full_map = mean_ap_metric(
                        self.all_preds, self.all_labels)[1]

                elif cfg.DATASET == 'ava':
                    (all_preds_arr, all_labels_arr, all_original_boxes_arr,
                     all_metadata_arr) = self.get_ava_eval_arr()

                    self.full_map = evaluate_ava(
                        all_preds_arr,
                        all_original_boxes_arr,
                        all_metadata_arr,
                        self.excluded_keys,
                        self.class_whitelist,
                        self.categories,
                        groundtruth=(self.full_groundtruth
                                          if cfg.AVA.FULL_EVAL
                                          else self.mini_groundtruth),
                        video_idx_to_name=self.video_idx_to_name,
                        name=name)
        else:
            self.avg_err = self.aggr_err / self.aggr_batch_size
            self.avg_err5 = self.aggr_err5 / self.aggr_batch_size

            if not is_train:
                all_preds, all_labels = self.stack_predictions()
                with open('epic_predictions_%s.pkl' % name, 'w') as f:
                    pickle.dump((all_preds, all_labels), f,
                                protocol=pickle.HIGHEST_PROTOCOL)

    def get_ava_eval_arr(self):
        """Stacking list of AVA predictions and labels into numpy arrays."""
        all_preds_arr = np.vstack(self.all_preds)
        all_labels_arr = np.vstack(self.all_labels)
        all_original_boxes_arr = np.vstack(self.all_original_boxes)
        all_metadata_arr = np.vstack(self.all_metadata)

        # If we use a fixed batch size, the final batch might contain additional
        # examples. We want to remove them and use only "self._total_num_boxes"
        # boxes.
        # The following is a sanity check to ensure we see and remove reasonable
        # amount of boxes. (Here we assume each example has < 50 boxes.)
        assert all_preds_arr.shape[0] >= self._total_num_boxes
        assert all_preds_arr.shape[0] - self._total_num_boxes \
            < cfg.TEST.BATCH_SIZE * 50
        assert all_preds_arr.shape[0] == all_labels_arr.shape[0]
        assert all_preds_arr.shape[0] == all_original_boxes_arr.shape[0]
        assert all_preds_arr.shape[0] == all_metadata_arr.shape[0]

        all_preds_arr = all_preds_arr[
            :self._total_num_boxes]
        all_labels_arr = all_labels_arr[
            :self._total_num_boxes]
        all_original_boxes_arr = all_original_boxes_arr[
            :self._total_num_boxes]
        all_metadata_arr = all_metadata_arr[
            :self._total_num_boxes]

        return (all_preds_arr, all_labels_arr, all_original_boxes_arr,
                all_metadata_arr)

    def get_computed_metrics(self):
        """Get testing summary."""
        json_stats = {}
        if cfg.MODEL.MULTI_LABEL:
            if self.split == 'train':
                json_stats['train_loss'] = self.avg_loss
                json_stats['train_full_map'] = self.full_map
            elif self.split in ['test', 'val']:
                json_stats['test_full_map'] = self.full_map
                json_stats['test_best_map'] = self.best_map
        else:
            if self.split == 'train':
                json_stats['train_loss'] = self.avg_loss
                json_stats['train_err'] = self.avg_err
                json_stats['train_err5'] = self.avg_err5

            elif self.split in ['test', 'val']:
                json_stats['test_err'] = self.avg_err
                json_stats['test_err5'] = self.avg_err5
                json_stats['best_err'] = self.best_top1
                json_stats['best_err5'] = self.best_top5

        return json_stats

    def log_final_metrics(self, model_iter, total_iters=None):
        """Print out final results."""
        if total_iters is None:
            total_iters = cfg.SOLVER.MAX_ITER
        if cfg.MODEL.MULTI_LABEL:

            info = ''
            if cfg.DATASET == 'ava':
                info = 'Box@%.5f ' % cfg.AVA.DETECTION_SCORE_THRESH

            print('* {} testing finished #iters [{}|{}]: mAP: {:.3f}'.format(
                info, model_iter + 1, total_iters, self.full_map))
        else:
            print(
                '* Finished #iters [{}|{}]: top1: {:.3f} top5: {:.3f}'.format(
                    model_iter + 1, total_iters,
                    100.0 - self.avg_err, 100.0 - self.avg_err5))

    def compute_and_log_best(self):
        """Log best model so far."""

        if cfg.MODEL.MULTI_LABEL:
            if self.full_map > self.best_map:
                self.best_map = self.full_map
                print('\n* Best model: mAP: {:7.3f}\n'.format(
                    self.best_map))
        else:
            if self.avg_err < self.best_top1:
                self.best_top1 = self.avg_err
                self.best_top5 = self.avg_err5
                print('\n* Best model: top1: {:7.3f} top5: {:7.3f}\n'.format(
                    self.best_top1, self.best_top5
                ))

    def calculate_and_log_all_metrics_train(
            self, curr_iter, timer, suffix=''):
        """Calculate and log metrics for training."""

        # To be safe, we always read lr from workspace.
        self.lr = float(
            workspace.FetchBlob('gpu_{}/lr'.format(cfg.ROOT_GPU_ID)))

        # To be safe, we only trust what we load from workspace.
        cur_batch_size = get_batch_size_from_workspace()

        # We only compute loss for train.
        # We multiply by cfg.MODEL.GRAD_ACCUM_PASS to calibrate.
        cur_loss = sum_multi_gpu_blob('loss')
        cur_loss = float(np.sum(cur_loss))

        self.aggr_loss += cur_loss * cur_batch_size
        self.aggr_batch_size += cur_batch_size

        if not cfg.MODEL.MULTI_LABEL:
            accuracy_metrics = compute_multi_gpu_topk_accuracy(
                top_k=1, split=self.split, suffix=suffix)
            accuracy5_metrics = compute_multi_gpu_topk_accuracy(
                top_k=5, split=self.split, suffix=suffix)

            cur_err = (1.0 - accuracy_metrics['topk_accuracy']) * 100
            cur_err5 = (1.0 - accuracy5_metrics['topk_accuracy']) * 100

            self.aggr_err += cur_err * cur_batch_size
            self.aggr_err5 += cur_err5 * cur_batch_size

        if (curr_iter + 1) % cfg.LOG_PERIOD == 0:
            rem_iters = cfg.SOLVER.MAX_ITER - curr_iter - 1
            eta_seconds = timer.average_time * rem_iters
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            epoch = (curr_iter + 1) \
                / (cfg.TRAIN.DATASET_SIZE / cfg.TRAIN.BATCH_SIZE)

            log_str = ' '.join((
                '| Train ETA: {} LR: {:.8f}',
                ' Iters [{}/{}]',
                '[{:.2f}ep]',
                ' Time {:0.3f}',
                ' Loss {:7.4f}',
            )).format(
                eta, self.lr,
                curr_iter + 1, cfg.SOLVER.MAX_ITER,
                epoch,
                timer.diff,
                cur_loss,
            )

            if not cfg.MODEL.MULTI_LABEL:
                log_str += ' top1 {:7.3f} top5 {:7.3f}'.format(
                    cur_err, cur_err5)
            print(log_str)

    def calculate_and_log_all_metrics_test(
            self, curr_iter, timer, total_iters, suffix=''):
        """Calculate and log metrics for testing."""

        # To be safe, we only trust what we load from workspace.
        cur_batch_size = get_batch_size_from_workspace()

        self.aggr_batch_size += cur_batch_size
        (preds, labels,
         original_boxes, metadata) = get_multi_gpu_outputs(self.split, suffix)

        self.all_preds.append(preds)
        self.all_labels.append(labels)
        if cfg.MODEL.MULTI_LABEL:
            self.all_original_boxes.append(original_boxes)
            self.all_metadata.append(metadata)
        else:
            accuracy_metrics = compute_multi_gpu_topk_accuracy(
                top_k=1, split=self.split, suffix=suffix)
            accuracy5_metrics = compute_multi_gpu_topk_accuracy(
                top_k=5, split=self.split, suffix=suffix)

            cur_err = (1.0 - accuracy_metrics['topk_accuracy']) * 100
            cur_err5 = (1.0 - accuracy5_metrics['topk_accuracy']) * 100

            self.aggr_err += cur_err * cur_batch_size
            self.aggr_err5 += cur_err5 * cur_batch_size

        if (curr_iter + 1) % cfg.LOG_PERIOD == 0 \
                or curr_iter + 1 == total_iters:

            test_str = ' '.join((
                '| Test: [{}/{}]',
                ' Time {:0.3f}',
                ' current batch {}',
                ' aggregated batch {}',
            )).format(
                curr_iter + 1, total_iters,
                timer.diff,
                cur_batch_size, self.aggr_batch_size
            )
            if not cfg.MODEL.MULTI_LABEL:
                test_str += (' top1 {:7.3f} ({:7.3f})'
                             + '  top5 {:7.3f} ({:7.3f})').format(
                    cur_err, self.aggr_err / self.aggr_batch_size,
                    cur_err5, self.aggr_err5 / self.aggr_batch_size,
                )
            print(test_str)


# ----------------------------------------------
# Other Utils
# ----------------------------------------------
def compute_topk_correct_hits(top_k, preds, labels):
    '''Compute the number of corret hits'''
    batch_size = preds.shape[0]

    top_k_preds = np.zeros((batch_size, top_k), dtype=np.float32)
    for i in range(batch_size):
        top_k_preds[i, :] = np.argsort(-preds[i, :])[:top_k]

    correctness = np.zeros(batch_size, dtype=np.int32)
    for i in range(batch_size):
        if labels[i] in top_k_preds[i, :].astype(np.int32).tolist():
            correctness[i] = 1
    correct_hits = sum(correctness)

    return correct_hits


def mean_ap_metric(predicts, targets):
    """Compute mAP, wAP, AUC for Charades."""

    predicts = np.vstack(predicts)
    targets = np.vstack(targets)
    logger.info(
        "Getting mAP for {} examples".format(
            predicts.shape[0]
        ))
    start_time = time.time()

    predict = predicts[:, ~np.all(targets == 0, axis=0)]
    target = targets[:, ~np.all(targets == 0, axis=0)]
    mean_auc = 0
    aps = [0]
    try:
        mean_auc = metrics.roc_auc_score(target, predict)
    except ValueError:
        print(
            'The roc_auc curve requires a sufficient number of classes \
            which are missing in this sample.'
        )
    try:
        aps = metrics.average_precision_score(target, predict, average=None)
    except ValueError:
        print(
            'Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample.'
        )

    mean_ap = np.mean(aps)
    weights = np.sum(target.astype(float), axis=0)
    weights /= np.sum(weights)
    mean_wap = np.sum(np.multiply(aps, weights))
    all_aps = np.zeros((1, targets.shape[1]))
    all_aps[:, ~np.all(targets == 0, axis=0)] = aps

    logger.info('\tDone in {} seconds'.format(time.time() - start_time))
    return mean_auc, mean_ap, mean_wap, all_aps.flatten()


def compute_multi_gpu_topk_accuracy(top_k, split, suffix='', epic_type=''):
    """Get predictions and labels from GPUs and compute top-k."""

    aggr_batch_size = 0
    aggr_top_k_correct_hits = 0

    computed_metrics = {}

    for idx in range(cfg.ROOT_GPU_ID, cfg.ROOT_GPU_ID + cfg.NUM_GPUS):
        softmax = workspace.FetchBlob('gpu_{}/pred{}'.format(idx, epic_type))

        softmax = softmax.reshape((softmax.shape[0], -1))
        labels = workspace.FetchBlob('gpu_{}/labels{}{}'.format(
            idx, epic_type, suffix))

        assert labels.shape[0] == softmax.shape[0], "Batch size mismatch."

        aggr_batch_size += labels.shape[0]

        aggr_top_k_correct_hits += compute_topk_correct_hits(
            top_k, softmax, labels)

    # Normalize results.
    computed_metrics['topk_accuracy'] = \
        float(aggr_top_k_correct_hits) / aggr_batch_size

    return computed_metrics


def get_multi_gpu_outputs(split, suffix):
    """Get predictions and labels from GPUs."""

    all_preds, all_labels = [], []
    all_original_boxes, all_metadata = [], []
    for idx in range(cfg.ROOT_GPU_ID, cfg.ROOT_GPU_ID + cfg.NUM_GPUS):

        softmax = workspace.FetchBlob('gpu_{}/pred'.format(idx))

        labels = workspace.FetchBlob('gpu_{}/labels{}'.format(idx, suffix))

        softmax = softmax.reshape((softmax.shape[0], -1))
        all_preds.append(softmax)
        all_labels.append(labels)

        if cfg.DATASET == 'ava':
            original_boxes = workspace.FetchBlob(
                'gpu_{}/original_boxes{}'.format(idx, suffix))
            metadata = workspace.FetchBlob('gpu_{}/metadata{}'.format(idx, suffix))

            all_original_boxes.append(original_boxes)
            all_metadata.append(metadata)

    if cfg.DATASET == 'ava':
        return (np.vstack(all_preds), np.vstack(all_labels),
                np.vstack(all_original_boxes), np.vstack(all_metadata))
    return np.vstack(all_preds),  np.vstack(all_labels), None, None


def sum_multi_gpu_blob(blob_name):
    """Sum values of a blob from all GPUs."""
    value = 0
    num_gpus = cfg.NUM_GPUS
    root_gpu_id = cfg.ROOT_GPU_ID
    for idx in range(root_gpu_id, root_gpu_id + num_gpus):
        value += workspace.FetchBlob('gpu_{}/{}'.format(idx, blob_name))
    return value


def get_batch_size_from_workspace():
    """Sum batch sizes of all GPUs."""
    value = 0
    num_gpus = cfg.NUM_GPUS
    root_gpu_id = cfg.ROOT_GPU_ID
    for idx in range(root_gpu_id, root_gpu_id + num_gpus):
        value += workspace.FetchBlob('gpu_{}/{}'.format(idx, 'pred')).shape[0]
    return value


# ----------------------------------------------
# For Logging
# ----------------------------------------------
def get_json_stats_dict(train_meter, test_meter, curr_iter):
    """Define stats that will be logged."""

    json_stats = {
        "eval_period" : cfg.TRAIN.EVAL_PERIOD,
        "batchSize" : cfg.TRAIN.BATCH_SIZE,
        "dataset" : cfg.DATASET,
        "num_classes" : cfg.MODEL.NUM_CLASSES,
        "momentum" : cfg.SOLVER.MOMENTUM,
        "weightDecay" : cfg.SOLVER.WEIGHT_DECAY,
        "nGPU" : cfg.NUM_GPUS,
        "LR" : cfg.SOLVER.BASE_LR,
        "bn_momentum" : cfg.MODEL.BN_MOMENTUM,
        "current_learning_rate" : train_meter.lr,
    }
    computed_train_metrics = train_meter.get_computed_metrics()
    json_stats.update(computed_train_metrics)
    if test_meter is not None:
        computed_test_metrics = test_meter.get_computed_metrics()
        json_stats.update(computed_test_metrics)

    # Other info.
    json_stats['used_gpu_memory'] = misc.get_gpu_stats()
    json_stats['currentIter'] = curr_iter + 1
    json_stats['epoch'] = \
        curr_iter / (cfg.TRAIN.DATASET_SIZE / cfg.TRAIN.BATCH_SIZE)

    return json_stats


# ----------------------------------------------
# AVA multi-crop testing utils.
# ----------------------------------------------
def combine_ava_multi_crops():
    """
    Multi-crop testing for AVA.
    It combines outputs of multiple scales, 2 flips, and 3 spatial shifts.
    """
    for threshold in cfg.AVA.DETECTION_SCORE_THRESH_EVAL:
        score_files = []
        for scale in cfg.AVA.TEST_MULTI_CROP_SCALES:
            for flip in [False, True]:
                shift_score_files = [
                    'detections_final_%d%s_shift%d_%.03f.csv' % (
                        scale, '_flip' if flip else '', shift, threshold)
                    for shift in range(3)]

                score_files.append(merge_ava_3shift_score_files(
                    shift_score_files, flip, scale))

        merge_ava_score_files(score_files)


def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def merge_ava_3shift_score_files(shift_score_files, flip, scale):
    """Load score files of 3 spatial shifts and combine them."""

    out_filename = shift_score_files[0].replace('_shift0', '_combined')
    video_shapes = {}
    logger.info("Combining scores of 3 spatial shifts:\n\t%s" %
        '\n\t'.join(shift_score_files))

    with open(shift_score_files[0], 'r') as fin0:
        with open(shift_score_files[1], 'r') as fin1:
            with open(shift_score_files[2], 'r') as fin2:

                with open(out_filename, 'w') as fout:
                    for line0, line1, line2 in zip(fin0, fin1, fin2):
                        items0 = line0.split(',')
                        items1 = line1.split(',')
                        items2 = line2.split(',')
                        score0 = float(items0[-1])
                        score1 = float(items1[-1])
                        score2 = float(items2[-1])

                        box = map(float, items0[2:6])
                        video = items0[0]
                        assert items0[0] == items1[0]
                        assert items0[0] == items2[0]

                        if video not in video_shapes:
                            im = cv2.imread(os.path.join(
                                cfg.DATADIR, video, video + '_000001.jpg'))
                            video_shapes[video] = im.shape

                        height, width, _ = video_shapes[video]
                        height, width = scale, float(width * scale) / height
                        norm_crop_size = float(min(height, 256)) / width

                        center_left = 0.5 - norm_crop_size / 2.0
                        center_right = 0.5 + norm_crop_size / 2.0
                        lcrop_right = norm_crop_size
                        rcrop_left = 1.0 - norm_crop_size

                        if flip:
                            box[0], box[2] = 1.0 - box[2], 1.0 - box[0]

                        # Note that an object might fall completely out of a
                        # crop. When merging spatial shifts, we discard
                        # predictions of crops that do not overlap with
                        # the object.
                        valid_scores = []
                        if box[2] > center_left and box[0] < center_right:
                            valid_scores.append(score1)
                        if box[0] < lcrop_right:
                            valid_scores.append(score0)
                        if box[2] > rcrop_left:
                            valid_scores.append(score2)
                        combined = float(np.mean(map(sigmoid, valid_scores)))

                        new_line = line0.split(',')[:-1]
                        new_line.append(str(combined))
                        new_line = ','.join(new_line)
                        fout.write(new_line + '\n')
    eval_ava_score_file(out_filename)
    return out_filename


def merge_ava_score_files(score_files):
    """
    Combine score files of different flips and scales and produce
    the final output.
    """

    out_filename = "final_multi_crop_testing_results.csv"
    logger.info("Combining scores of multiple scales and flips:\n\t%s" %
        '\n\t'.join(score_files))

    all_lines = []
    for score_file in score_files:
        with open(score_file, 'r') as f:
            all_lines.append(f.readlines())

    with open(out_filename, 'w') as fout:
        for s_lines in zip(*all_lines):
            combined = float(np.sum(
                [float(s_line.split(',')[-1]) for s_line in s_lines]))
            new_line = s_lines[0].split(',')[:-1]

            new_line.append('%f' % combined)
            new_line = ','.join(new_line)
            fout.write(new_line + '\n')
    eval_ava_score_file(out_filename)


def eval_ava_score_file(score_filename):
    """Evaluate AVA given files (as opposed to given numpy arrays)."""
    evaluate_ava_from_files(
        os.path.join(cfg.AVA.ANNOTATION_DIR,
            "ava_action_list_v2.1_for_activitynet_2018.pbtxt"),
        os.path.join(cfg.AVA.ANNOTATION_DIR,
            "ava_val_v2.1.csv"),
        score_filename,
        os.path.join(cfg.AVA.ANNOTATION_DIR,
            "ava_val_excluded_timestamps_v2.1.csv"))
