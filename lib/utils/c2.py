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

"""Helpful utilities for working with Caffe2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

from caffe2.python import dyndep


# Default value of the CMake install prefix
_CMAKE_INSTALL_PREFIX = '/usr/local'
_DETECTRON_OPS_LIB = 'libcaffe2_detectron_ops_gpu.so'


def get_detectron_ops_lib():
    """Retrieve Detectron ops library."""
    # Candidate prefixes for detectron ops lib path.
    prefixes = [_CMAKE_INSTALL_PREFIX, sys.prefix, sys.exec_prefix] + sys.path

    # Candidate subdirs for detectron ops lib.
    subdirs = ['lib', 'torch/lib']

    # Try to find detectron ops lib.
    for prefix in prefixes:
        for subdir in subdirs:
            ops_path = os.path.join(prefix, subdir, _DETECTRON_OPS_LIB)
            if os.path.exists(ops_path):
                print('Found Detectron ops lib: {}'.format(ops_path))
                return ops_path
    raise Exception('Detectron ops lib not found')


def import_detectron_ops():
    """Import Detectron ops."""
    detectron_ops_lib = get_detectron_ops_lib()
    dyndep.InitOpsLibrary(detectron_ops_lib)
