# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Util to invoke tarball in the system."""
# pylint: disable=invalid-name
from __future__ import absolute_import as _abs
import os
import shutil
import subprocess
from . import utils
from .._ffi.base import py_str


def stackvm(output, files):
    """Simply copy StackVM output to the destination.

    Parameters
    ----------
    output : str
        The target StackVM file.

    files : list
        A single self-contained StackVM module file.
    """

    print(output, files)

    if len(files) == 0:
        raise RuntimeError("StackVM artifact must be provided")
    if len(files) > 1:
        "Unexpected multiple StackVM artifacts"

    shutil.copy(files[0], output)

# assign output format
stackvm.output_format = "stackvm"
