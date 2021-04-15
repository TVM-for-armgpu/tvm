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
"""Testcode for Android RPC.

To use it, start an RPC tracker with "python -m tvm.exec.rpc_tracker".
Use the tracker's address and port when configuring the RPC app.
Use "Adreno640" as the key if you wish to avoid modifying this script.
"""

import tvm
from tvm import te
import os
from tvm import rpc
from tvm.contrib import utils, ndk
import numpy as np
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python

# Set to be address of tvm proxy.
tracker_host = '127.0.0.1'
tracker_port = 9090
key = "MaliG76"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch

# whether enable to execute test on OpenCL target
test_opencl = True 
# whether enable to execute test on Vulkan target
test_vulkan = False


def _convert_to_remote(func, remote):
    """ convert module function to remote rpc function"""
    temp = utils.tempdir()
    path_dso = temp.relpath("tmp_func.so")
    func.export_library(path_dso, ndk.create_shared)

    remote.upload(path_dso)
    func = remote.load_module("tmp_func.so")
    return func



def measure_compute_mad(
    total_item, item_per_thread, base_type, bits, lanes, target, target_host, remote, ctx, n_times
):
    """measure peak compute speed by computing mad for a type

    The IR for measurement is

    for each thread
        for i in 1..item_per_thread
            x = mad(x, x, y)
            y = mad(y, y, x)

    Parameters
    ----------
    total_item: int
        number of elements in input array
    item_per_thread: int
        number of operations each thread does
    base_type: str
        can be "int", "float"
    bits: int
        can be 16, 32
    lanes: int
       lane of the vector type, can be 1, 2, 4, 8, 16
    target: :any:`tvm.target.Target`
        the target and option of the compilation.
    target_host : str or :any:`tvm.target.Target`
        host compilation target
    remote: tvm.rpc.RPCSession
        if it is not None, use remote rpc session
    ctx: TVMcontext
        the context of array
    n_times: int
        number of runs for taking mean

    Returns
    -------
    GOPS: float
         giga operation per second
    """

    n = total_item

    if bits >= 64 or lanes >= 16:
        n //= 2

    max_threads = target.max_num_threads

    base_type = str(base_type) + str(bits)
    dtype = base_type if lanes == 1 else base_type + "x" + str(lanes)

    def extern(ins, outs):
        # pylint: disable=unused-argument
        """construct measurement function by building IR directly"""
        ib = tvm.tir.ir_builder.create()

        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")

        ib.scope_attr(bx, "thread_extent", n // max_threads)
        ib.scope_attr(tx, "thread_extent", max_threads)

        idx = bx.var * max_threads + tx.var

        a = ib.allocate(dtype, (1), name="a", scope="local")
        b = ib.allocate(dtype, (1), name="b", scope="local")

        a[0] = outs[0].vload(idx, dtype)
        b[0] = outs[0].vload(idx, dtype)

        if base_type.find("float") != -1:

            def mad_func(x, y):
                return x * x + y

        else:

            def mad_func(x, y):
                return y * y + x

        for _ in range(item_per_thread // 4 // lanes):
            a[0] = mad_func(a[0], b[0])
            b[0] = mad_func(b[0], a[0])

        ib.emit(outs[0].vstore(idx, b[0]))
        return ib.get()

    y = te.extern((n,), [], extern, name="y", dtype=dtype)
    s = te.create_schedule(y.op)

    try:
        func = tvm.build(s, [y], target, target_host=target_host)
        #print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
        func = _convert_to_remote(func, remote)
        time_f = func.time_evaluator(func.entry_name, ctx, number=n_times)
        y = tvm.nd.empty((n,), dtype=dtype, ctx=ctx)
        time = time_f(y).mean
    except tvm._ffi.base.TVMError:
        # build error (occur when device does not support half)
        return -1

    return 1.0 * (n * item_per_thread) / 1e9 / time


def measure_compute_all_types(
    total_item, item_per_thread, n_times, target, target_host, remote, ctx, verbose=True
):
    """measure peak flops for all types

    Parameters
    ----------
    total_item: int
        number of elements in input array
    item_per_thread: int
        number of elements each thread accmulates
    n_times: int
        number of runs for averaging
    target: :any:`tvm.target.Target`
        the target and option of the compilation.
    target_host : str or :any:`tvm.target.Target`
        host compilation target
    remote: tvm.rpc.RPCSession
        remote rpc session
    ctx: TVMcontext
        the context of array
    verbose: bool
        whether outputs immediate result

    Returns
    -------
    result: list
        a list of (type_name, GFLOPS/GIOPS) pairs
    """
    result = []
    for base_type in ["float", "int"]:
        for bits in [16, 32]:
            for lanes in [1, 2, 4, 8, 16]:
                if base_type == "int" and bits != 32:  # only measure int32
                    continue

                max_speed = -1e9
                for per_thread in [item_per_thread // 2, item_per_thread, item_per_thread * 2]:
                    speed = measure_compute_mad(
                        total_item,
                        per_thread,
                        base_type,
                        bits,
                        lanes,
                        target,
                        target_host,
                        remote,
                        ctx,
                        n_times,
                    )
                    max_speed = max(max_speed, speed)
                type_name = base_type + str(bits)
                result.append(["%sx%d" % (type_name, lanes), max_speed])

                unit = "GFLOPS" if base_type == "float" else "GIOPS"

                if verbose:
                    import logging
                    logging.warning("\t%-10s %.2f %s", result[-1][0], result[-1][1], unit)

    return result




def test_rpc_module():
    # graph
    open_image=1
    ddtype='float32'
    if open_image == 1:
        ddtype = "climgfloatr32"
    temp = utils.tempdir()

    ## Establish remote connection with target hardware
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    remote = tracker.request(key, priority=0, session_timeout=60)

    # Compile the Graph for OpenCL target
    if test_opencl:
        N, H, W, CO, CI, KH, KW, strides, padding = 1, 64, 64, 512, 256, 1, 1, (1, 1), (0, 0)
        PACK4 = 4
        W_P = H
        H_P = H
        C_P = CI//PACK4
        K_P = CO//PACK4
        n_times = 20
        target = tvm.target.Target("opencl")
        ctx = remote.context(str(target), 0)
        bandwidth_total_item = 1 << 25
        bandwidth_item_per_thread = 32
        compute_total_item = 1 << 21
        compute_item_per_thread = 4096
        measure_compute_all_types(
            compute_total_item, compute_item_per_thread, n_times, target, target_host, remote, ctx
        )
        return


if __name__ == "__main__":
    test_rpc_module()
