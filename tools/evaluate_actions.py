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
Combine EPIC-Kitchens verb and noun predictions to get "action" predictions,
and compute top-k accuracy.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cPickle as pkl
import csv
import logging
import numpy as np
import os
import sys

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


NUM_TEST_SEG = 5281


def get_training_action_freq(num_verbs, num_nouns, annotation_root):
    """Estimating the frequency of each (verb, noun) pair in training set."""
    seen = np.zeros((num_verbs, num_nouns))

    with open(os.path.join(annotation_root,
                           'EPIC_train_action_labels.csv'), 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            assert len(row) == 14, len(row)
            person = int(row[1][1:])
            assert person >= 1 and person <= 32

            # If in training split.
            if person <= 25:
                seen[int(row[-5]), int(row[-3])] += 1

    return seen / seen.sum()


def compute_top_k_verbs_or_nouns(scores, labels, K):
    """Compute top-k accuracy for verbs or nouns."""
    assert NUM_TEST_SEG == scores.shape[0]
    assert NUM_TEST_SEG == labels.shape[0]

    correct_count = 0
    for i in range(NUM_TEST_SEG):
        if int(labels[i]) in scores[i].argsort()[-K:]:
            correct_count += 1

    accuracy = 100.0 * float(correct_count) / NUM_TEST_SEG
    logger.info('Top-%d: %.04f%%' % (K, accuracy))


def compute_top_k_actions(
        verb_pred, noun_pred, verb_labels, noun_labels, K, prior=None):
    """Compute top-k accuracy for actions."""
    assert NUM_TEST_SEG == verb_pred.shape[0]
    assert NUM_TEST_SEG == noun_pred.shape[0]

    assert NUM_TEST_SEG == verb_labels.shape[0]
    assert NUM_TEST_SEG == noun_labels.shape[0]

    correctness = np.zeros(NUM_TEST_SEG, dtype=np.int32)
    for i in range(NUM_TEST_SEG):
        action_scores = np.outer(verb_pred[i, :], noun_pred[i, :])

        if prior is not None:
            action_scores *= prior
        top_verbs, top_nouns = np.unravel_index(
            np.argsort(-action_scores, axis=None), action_scores.shape)

        for cur_v, cur_n in zip(top_verbs[:K].tolist(), top_nouns[:K].tolist()):
            if int(verb_labels[i]) == cur_v and int(noun_labels[i]) == cur_n:
                correctness[i] = 1

    accuracy = 100.0 * float(sum(correctness)) / NUM_TEST_SEG
    logger.info('Top-%d: %.04f%%' % (K, accuracy))


def softmax(x):
    """Row-wise softmax given a 2D matrix."""
    assert len(x.shape) == 2
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def evaluate_actions(args):
    """Evaluate performance on "Actions" given "Verb" and "Noun" predictions."""
    with open(args.verb_file, 'rb') as f:
        verb_pred, verb_labels = pkl.load(f)

    with open(args.noun_file, 'rb') as f:
        noun_pred, noun_labels = pkl.load(f)

    verb_pred = softmax(verb_pred)
    noun_pred = softmax(noun_pred)

    assert verb_pred.shape[0] == NUM_TEST_SEG
    assert verb_labels.shape[0] == NUM_TEST_SEG

    assert noun_pred.shape[0] == NUM_TEST_SEG
    assert noun_labels.shape[0] == NUM_TEST_SEG

    action_freq = get_training_action_freq(
        num_verbs=verb_pred.shape[1],
        num_nouns=noun_pred.shape[1],
        annotation_root=args.annotation_root)

    v_given_n = action_freq \
        / (np.sum(action_freq, axis=1, keepdims=True) + 1e-5)

    for K in [1, 5]:
        logger.info("Verbs:")
        compute_top_k_verbs_or_nouns(verb_pred, verb_labels, K)
        logger.info("Nouns:")
        compute_top_k_verbs_or_nouns(noun_pred, noun_labels, K)
        logger.info("Actions:")
        compute_top_k_actions(
            verb_pred, noun_pred, verb_labels, noun_labels, K, v_given_n)


def main():
    parser = argparse.ArgumentParser(
        description='EPIC-Kitchens Action Evaluation')
    parser.add_argument(
        '--verb_file', type=str, required=True, help='Verb prediction results.')
    parser.add_argument(
        '--noun_file', type=str, required=True, help='Noun prediction results.')
    parser.add_argument(
        '--annotation_root', type=str, default='data/epic/annotations',
        help='Path to EPIC-Kitchens annotation folder.')

    args = parser.parse_args()
    evaluate_actions(args)


if __name__ == '__main__':
    main()
