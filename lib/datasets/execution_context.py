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
This class creates a shared memory buffer that can be used during multiprocessing
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import logging
import ctypes
from multiprocessing.sharedctypes import RawArray
from multiprocessing import Pool

logger = logging.getLogger(__name__)


class ExecutionContext(object):

    def __init__(self, data_size, worker_ids, batch_size, num_processes):
        # the execution context should have the worker ids, pools,
        self._expected_data_size = data_size
        self._worker_ids = worker_ids
        self.batch_size = batch_size
        self._num_processes = num_processes
        self.pools = None
        self.shared_data_lists = None
        self.create_execution_context()

    def create_execution_context(self):
        pools = {}
        shared_data_lists = {}
        for worker_id in self._worker_ids:
            shared_data_list = []
            shared_data_lists[worker_id] = shared_data_list

            # for each worker_id, we fetch a batch size of 32 and this is being
            # done by various parallel processes
            for _ in range(self.batch_size):
                shared_arr = RawArray(ctypes.c_float, self._expected_data_size)
                shared_data_list.append(shared_arr)

            pools[worker_id] = Pool(
                processes=self._num_processes,
                initializer=self._init_pool,
                initargs=(
                    shared_data_list,
                )
            )
        self.pools = pools
        self.shared_data_lists = shared_data_lists
        logger.info('execution_context created...')
        logger.info('pools: {}'.format(pools))
        logger.info('shared_data_lists: {}'.format(shared_data_lists))

    def _init_pool(self, data_list):
        """
        Each pool process calls this initializer.
        Load the array to be populated into that process's global namespace.
        """
        global shared_data_list
        shared_data_list = data_list

    def _shutdown_pools(self):
        pools = self.pools
        logger.info('Shutting down multiprocessing pools')
        for i, p in enumerate(pools.values()):
            logger.info('shutting pool: {}'.format(i))
            try:
                p.close()
                p.join()
            except Exception as e:
                logger.info('Exception when closing pool: {}'.format(e))
                continue
        logger.info('Finished multi-processing data loading')
