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

"""This script is responsible for loading data for each GPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
import cProfile
import logging
import numpy as np
import pstats
import Queue
import random
import signal
import StringIO
import threading
import time
import uuid

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, scope
from core.config import config as cfg
from datasets.coordinator import Coordinator, coordinated_put, coordinated_get

from datasets.ava import AvaDataset
from datasets.ava_data_input import create_data_input \
    as create_ava_data_input

from datasets.charades import CharadesDataset
from datasets.charades_data_input import create_data_input \
    as create_charades_data_input

from datasets.epic import EpicDataset
from datasets.epic_data_input import create_data_input as \
    create_epic_data_input


logger = logging.getLogger(__name__)
db_loader_map = {
    'charades': CharadesDataset,
    'ava': AvaDataset,
    'epic': EpicDataset,
}
create_data_input_map = {
    'charades': create_charades_data_input,
    'ava': create_ava_data_input,
    'epic': create_epic_data_input,
}


class DataLoader(object):

    def __init__(
        self,
        split,
        input_db,
        batch_size,
        num_workers=4,
        num_processes=12,
        minibatch_queue_size=64,
        blobs_queue_capacity=1,
        node_id=0,
        loader_stats_file=None,
        suffix='',
        crop_size=224,
    ):

        # Debugging tool.
        self._loader_stats_file = loader_stats_file
        if loader_stats_file is not None:
            logger.info("Profiling minibatch loader and saving to {}".format(
                        loader_stats_file))

        self._node_id = node_id
        self._split = split
        self._input_db = input_db
        self._db_size = input_db.get_db_size()
        self._lock = threading.Lock()
        self._batch_size = batch_size
        self._current = 0
        self._perm = np.arange(input_db.get_db_size())

        self.coordinator = Coordinator()
        self._num_gpus = cfg.NUM_GPUS

        self._num_workers = num_workers
        self._num_processes = num_processes

        self._crop_size = crop_size
        self._expected_data_size = 3 * self._crop_size ** 2

        self._minibatch_queue_capacity = minibatch_queue_size
        self._minibatch_queue = Queue.Queue(maxsize=minibatch_queue_size)
        self._gpu_blobs_queue_capacity = blobs_queue_capacity
        self._blobs_queue_name = '{}_blobs_queue_{}'.format(
            cfg.DATASET, str(uuid.uuid4())
        )

        self.suffix = suffix
        self.blobnames = input_db.blobnames

        # Assign indexes to blobs so that they can be queued in the same order.
        self._blobs_idx_map = OrderedDict()
        for i, blobname in enumerate(self.blobnames):
            self._blobs_idx_map[blobname] = i

        if split == 'train':
            self._shuffle_db_indices(self._db_size)

        self._mb_index = 0
        self._enqueue_mb_index = 0
        self._enqueue_ooo_buf = {}
        self.create_threads()

        self._create_data_input()

    def get_worker_ids(self):
        if self._split == 'train':
            return range(0, self._num_workers)
        else:
            assert self._num_workers < 100
            return range(100, 100 + self._num_workers)

    def _create_data_input(self):
        create_data_input = create_data_input_map[cfg.DATASET]

        (context_execution, fetch_func) = create_data_input(
            self._input_db, self._expected_data_size, self._num_processes,
            self._num_workers, self._split, self._batch_size,
            crop_size=self._crop_size,
        )
        self._context_execution = context_execution
        self._minibatch_fetch_func = fetch_func
        worker_ids = self.get_worker_ids()
        self._context_execution(worker_ids)

    def get_blob_names(self):
        return self._blobs_idx_map.keys()

    def create_blobs_queue(self, queue_name, num_blobs, capacity):
        """
        Create a BlobsQueue in the workspace to hold the mini-batches. Each GPU
        has its own workspace and we chose the namescope per GPU.
        """
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'CreateBlobsQueue',
                [], [queue_name],
                num_blobs=num_blobs,
                capacity=capacity,
            )
        )

    def close_blobs_queue(self):
        """Close a BlobsQueue"""
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'CloseBlobsQueue',
                [self._blobs_queue_name],
                []
            )
        )

    def _shuffle_db_indices(self, db_size):
        """Randomly permute the training roidb"""

        assert(self._split == 'train')

        indices = range(db_size)
        random.shuffle(indices)
        self._perm = indices
        self._current = 0
        return None


    def _get_next_minibatch_indices(self):
        """
        For single machine training: data can be randomly shuffled in K bins
        For distributed training: data can be either:
            (i) randomly sampled
            (ii) sampled from a global shuffle permutations
        """
        db_size = self._db_size
        with self._lock:
            mb_index = self._mb_index
            self._mb_index += 1
            if self._split == 'train':
                if ((self._current + self._batch_size) >= db_size):
                    self._shuffle_db_indices(db_size)

                db_indices = self._perm[
                    self._current:self._current + self._batch_size
                ]
                self._current += self._batch_size
                return db_indices, mb_index
            elif self._split in ['test', 'val']:
                if self._current == db_size:
                    self._current = 0
                elif self._current > db_size:
                    self._current = 0

                end_idx = self._current + self._batch_size
                db_indices = self._perm[self._current:end_idx]
                self._current += self._batch_size
                return db_indices, mb_index

    def _get_next_minibatch(self, worker_id):
        """
        Returns next blobs to be used for the next mini-batch queue
        """
        db_indices, mb_index = self._get_next_minibatch_indices()
        blobs = self._minibatch_fetch_func(
            self._input_db, worker_id, self._batch_size, db_indices,
            self._crop_size,
        )

        assert len(self.blobnames) == len(blobs), \
            'Expected %d blobs; got %d blobs' % (
                len(self.blobnames), len(blobs))
        minibatch_blobs = {
            name: blob for name, blob in zip(self.blobnames, blobs)
        }
        return minibatch_blobs, mb_index

    def minibatch_loader(self, worker_id):
        """Load mini-batches and put them into a queue in CPU memory"""
        if self._loader_stats_file is not None:
            prof = cProfile.Profile()
            prof.enable()
        with self.coordinator.stop_on_execution():
            while not self.coordinator.should_stop():
                minibatch_blobs, mb_index = self._get_next_minibatch(worker_id)
                ordered_minibatch_blobs = OrderedDict()
                for key in self.get_blob_names():
                    ordered_minibatch_blobs[key] = minibatch_blobs[key]
                coordinated_put(
                    self.coordinator,
                    self._minibatch_queue,
                    (mb_index, ordered_minibatch_blobs),
                )
        if self._loader_stats_file is not None:
            prof.disable()
            s = StringIO.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
            ps.print_stats()
            with open(self._loader_stats_file, 'w') as f:
                f.write(s.getvalue())
        logger.debug("Stopping mini-batch loader thread...")

    def enqueue_blobs(
        self,
        gpu_id,
        enqueue_blobs_names,
        blob_values,
    ):
        enqueue_blobs_names = [
            'gpu_{}/{}'.format(
                gpu_id, enqueue_blob_name
            ) for enqueue_blob_name in enqueue_blobs_names
        ]

        deviceOption = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
        for (blob_name, blob) in zip(enqueue_blobs_names, blob_values):
            workspace.FeedBlob(blob_name, blob, device_option=deviceOption)

        queue_name = 'gpu_{}/{}'.format(gpu_id, self._blobs_queue_name)
        workspace.RunOperatorOnce(
            core.CreateOperator(
                'SafeEnqueueBlobs',
                [queue_name] + enqueue_blobs_names,
                enqueue_blobs_names + [queue_name + '_enqueue_status'],
                device_option=deviceOption,
            )
        )

    def enqueue_blobs_thread(self, _gpu_id, enqueue_blobs_names):
        """
        Transfer mini-batches from the CPU mini-batch queue to all GPU
        BlobsQueues.
        """
        with self.coordinator.stop_on_execution():
            while not self.coordinator.should_stop():
                root_gpu_id = cfg.ROOT_GPU_ID
                for gpu_id in range(root_gpu_id, root_gpu_id + self._num_gpus):
                    if self._enqueue_mb_index in self._enqueue_ooo_buf:
                        blobs = self._enqueue_ooo_buf[self._enqueue_mb_index]
                        del self._enqueue_ooo_buf[self._enqueue_mb_index]
                    else:
                        while True:
                            (mb_index, blobs) = coordinated_get(
                                self.coordinator, self._minibatch_queue
                            )
                            if self._enqueue_mb_index == mb_index:
                                break
                            else:
                                self._enqueue_ooo_buf[mb_index] = blobs
                    self.enqueue_blobs(
                        gpu_id,
                        enqueue_blobs_names,
                        blobs.values(),
                    )
                    self._enqueue_mb_index += 1
        logger.debug("Stopping enqueuer thread...")

    # Minibatch loader threads: each thread builds minibatches and places them
    # into a queue in CPU memory.
    def create_threads(self):
        # "worker" threads to construct (partial) minibatches and put them on
        # minibatch queue in CPU memory (limited by queue size).
        self._worker_ids = self.get_worker_ids()
        self._workers = [
            threading.Thread(
                target=self.minibatch_loader,
                name='worker_{}'.format(worker_id),
                args=[worker_id],
            ) for worker_id in self._worker_ids
        ]

        # Create one BlobsQueue per GPU which holds the training data in GPU
        # memory and feeds to the net.
        root_gpu_id = cfg.ROOT_GPU_ID
        for gpu_id in range(root_gpu_id, root_gpu_id + self._num_gpus):
            with core.NameScope('gpu_{}'.format(gpu_id)):
                self.create_blobs_queue(
                    queue_name=self._blobs_queue_name,
                    num_blobs=len(self._blobs_idx_map),
                    capacity=self._gpu_blobs_queue_capacity
                )

        # Launch enqueuer threads.
        blob_names = self._blobs_idx_map.keys()
        enqueue_blobs_names = [
            '{}_{}_enqueue'.format(self._split, blob_name)
            for blob_name in blob_names
        ]
        for gpu_id in range(root_gpu_id, root_gpu_id + self._num_gpus):
            with core.NameScope('gpu_{}'.format(gpu_id)):
                with core.DeviceScope(
                    core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
                ):
                    for blob_list in enqueue_blobs_names:
                        for blob in blob_list:
                            scoped_blob_name = scope.CurrentNameScope() + blob
                            workspace.CreateBlob(scoped_blob_name)
        self._enqueuer = threading.Thread(
            target=self.enqueue_blobs_thread, args=(0, enqueue_blobs_names)
        )

    def prefill_minibatch_queue(self):
        logger.info('Pre-filling {} minibatch queue'.format(self._split))
        while(self.minibatch_queue_size() < self._minibatch_queue_capacity):
            time.sleep(1.0)
        logger.info("{} minibatch queue pre-filled.".format(self._split))

    def start(self, prefill=False):
        for w in self._workers + [self._enqueuer]:
            w.daemon = True
            w.start()
        if prefill:
            self.prefill_minibatch_queue()

    def join(self):
        for w in self._workers + [self._enqueuer]:
            w.join()

    def shutdown_dataloader(self):
        self.coordinator.request_stop()
        self.coordinator.wait_for_stop()
        root_gpu_id = cfg.ROOT_GPU_ID
        for idx in range(root_gpu_id, root_gpu_id + self._num_gpus):
            with core.NameScope("gpu_{}".format(idx)):
                self.close_blobs_queue()
        self.join()

    def register_sigint_handler(self):
        def signal_handler(signal, frame):
            logger.info(
                "SIGINT: shutting down data loader threads and exiting")
            self.shutdown_dataloader()
        signal.signal(signal.SIGINT, signal_handler)

    def minibatch_queue_size(self):
        return self._minibatch_queue.qsize()


def get_input_db(dataset, data_type, model,
                 lfb_infer_only=False,
                 shift=None, lfb=None, suffix=''):
    assert dataset in db_loader_map.keys(), \
        "Unknown dataset: {}".format(dataset)

    input_db = db_loader_map[dataset](
        split=data_type,
        lfb_infer_only=lfb_infer_only,
        shift=shift, lfb=lfb, suffix=suffix)

    return input_db
