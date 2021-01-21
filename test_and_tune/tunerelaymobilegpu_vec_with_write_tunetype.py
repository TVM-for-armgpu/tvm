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
"""
Auto-tuning a Convolutional Network for Mobile GPU
==================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Eddie Yan <https://github.com/eqy>`_

Auto-tuning for a specific device is critical for getting the best
performance. This is a tutorial about how to tune a whole convolutional
network.

The operator implementation for Mobile GPU in TVM is written in template form.
The template has many tunable knobs (tile factor, vectorization, unrolling, etc).
We will tune all convolution, depthwise convolution and dense operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the TVM compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some arm devices. You can go to
`Mobile GPU Benchmark <https://github.com/apache/tvm/wiki/Benchmark#mobile-gpu>`_
to see the results.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import os

import numpy as np

import tvm
from tvm import te
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python

import logging
import sys
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

TRACKER_PORT=9090
open_image=1
@autotvm.template("tutorial/conv2d_no_batching")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"
    
    cfg = autotvm.get_config()

    #cfg.define_knob("idtype",[0,1])
    cfg.define_knob("kdtype",[0,1])
    #cfg.define_knob("wdtype",[0,2])
    typedict = {0:"float32",1:"climgfloatr32",2:"climgfloatw32"}

    A = te.placeholder((N, CI, H, W), name="data", dtype="climgfloatr32")
    W = te.placeholder((CO, CI, KH, KW), name="filter",dtype="climgfloatr32")
    B = topi.nn.conv2d_nchw(A, W, stride, padding, dilation=0, out_dtype="climgfloatw32")
    B.dtype == "climgfloatw32"


    s = te.create_schedule([B.op])

    ##### space definition begin #####
    n, f, y, x = s[B].op.axis
    rc, ry, rx = s[B].op.reduce_axis

    cfg = autotvm.get_config()
    #cfg.define_split("tile_f", f, num_outputs=4)
    #cfg.define_split("tile_y", y, num_outputs=4)
    #cfg.define_split("tile_x", x, num_outputs=4)
    #cfg.define_split("tile_rc", rc, num_outputs=3)
    #cfg.define_split("tile_ry", ry, num_outputs=3)
    #cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0,256, 512])
    cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # inline padding
    Apad = s[B].op.input_tensors[0]

    # Designate the memory hierarchy
    s = te.create_schedule(B.op)
    s[Apad].compute_inline()  # compute Apad inline
    AA = s.cache_read(Apad, "shared", [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    ###############################################################################
    # Blocking
    # --------
    #
    # The following code splits the workload into thread blocks and individual
    # threads. We follow the blocking scheme in the matrix multiply. As shown in the
    # figure below, given a pixel coordinate (y, x), a thread block is responsible
    # for computing a region of block_factor x block_factor (64 x 64) for output
    # channels and batch. Due to the limit of shared memory space, we only load step
    # x block_factor (8 x 64) data from Apad and B each time to buffers in the
    # shared memory.
    #
    # .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/conv_gpu_blocking.png
    #      :align: center
    #      :height: 308px
    #      :width: 317px
    #

    # tile consts
    tile = 1
    num_thread = 16
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
    thread_z = te.thread_axis((0, num_thread), "threadIdx.z")
    thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

    # Split the workloads
    hi, wi, fi, ni = s[B].op.axis
    kernel_scope = ni  # this is the scope to attach global config inside this kernel
    #bz = s[B].fuse(fi,ni)
    bz, fi = s[B].split(fi, factor=block_factor)
    bx, hi = s[B].split(hi, factor=block_factor)
    by, wi = s[B].split(wi, factor=block_factor)
    # Bind the iteration variables to GPU thread indices
    s[B].bind(bz, block_z)
    s[B].bind(by, block_y)
    s[B].bind(bx, block_x)

    ###############################################################################
    # Virtual Thread Split
    # --------------------
    #
    # We further split the workload from a thread block to individual threads. To
    # avoid *memory bank conflict*, we use virtual thread to split the area into 4
    # parts, and then tile into 8x8 grids. Therefore, shown in the figure below,
    # each thread computes 4 strided grids, where size of each grid is 4 x 4.
    #
    # .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/conv_gpu_vthread.png
    #      :align: center
    #      :height: 188px
    #      :width: 268px
    #

    #tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    #txz, wi = s[B].split(wi, nparts=vthread)  # virtual thread split
    ty, fi = s[B].split(fi, nparts=num_thread)
    tx, wi = s[B].split(wi, nparts=num_thread)
    tz, ni = s[B].split(ni, nparts=num_thread)
    #s[B].reorder(bz, by, bx, tyz, txz, ty, fi, ni)
    #s[B].reorder(bz,by,bx,wi, fi,ty)
    s[B].reorder(bz,by,bx,ty,wi, fi)

    #s[B].bind(tyz, thread_yz)
    #s[B].bind(txz, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)

    ###############################################################################
    # Cooperative Fetching
    # --------------------
    #
    # As mentioned before, each time step we need to transfer step x block_factor
    # data from GPU global memory to shared memory. In order to reduce the memory
    # transfer per thread, the following code lets threads in the same thread block
    # coopertively fetch dependent data from global memory.
    #


    # Schedule BL local write
    s[BL].compute_at(s[B],ni)
    yi, xi, fi, ni = s[BL].op.axis
    ry, rx, rc = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)
    s[BL].reorder(rco, ry, rx, rci, fi, ni)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rx)
    s[WW].compute_at(s[BL], rx)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    yi, xi, ci, ni = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ci = s[AA].split(ci, factor=4)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(ci)  # vectorize memory load

    # Schedule for W's shared memory load
    #yi, xi, ci, fi = s[WW].op.axis
    yi,xi,ci,fi=s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=num_thread)
    fi, fv = s[WW].split(fi, factor=4)
    tx, fi = s[WW].split(fi, nparts=num_thread)
    s[WW].reorder(ty, tx, yi, xi, ci, fi)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)
    s[WW].vectorize(fv)  # vectorize memory load

    # tune unroll
    s[B].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[B].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    return s, [A,W, B]

# ------------------
# Before tuning, we should apply some configurations. Here I use an RK3399 board
# as example. In your setting, you should modify the target and device_key accordingly.
# set :code:`use_android` to True if you use android phone.

#### DEVICE CONFIG ####

target = tvm.target.Target("opencl")
#target = tvm.target.Target("opencl -device=mali")

# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target host is used for cross compilation. You can query it by :code:`gcc -v` on your device.
#target_host = "llvm -mtriple=aarch64-linux-gnu"
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch

# Also replace this with the device key in your tracker
device_key = "android"

# Set this to True if you use android phone
use_android = True

#### TUNING OPTION ####
network = "tunetype"
log_file = "%s.%s.log" % (device_key, network)
dtype = "float32"

tuning_option = {
    "log_filename": log_file,
    "tuner": "xgb",
    "n_trial": 2000,
    "use_transfer_learning":True,
    "early_stopping": 450,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
        runner=autotvm.RPCRunner(
            device_key,
            host="0.0.0.0",
            port=TRACKER_PORT,
            number=10,
            timeout=5,
        ),
    ),
}

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default values provided here work well.
#   If you have enough time budget, you can set :code:`n_trial`, :code:`early_stopping` larger,
#   which makes the tuning run longer.
#   If your device runs very slow or your conv2d operators have many GFLOPs, considering to
#   set timeout larger.
#

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    use_transfer_learning=True
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    #if os.path.exists(tmp_log_file):
    #    os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                print("load previous results")
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    #os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    #mod, params, input_shape, _ = get_network(network, batch_size=1)
    #tasks = autotvm.task.extract_from_program(
    #    mod["main"],
    #    target=target,
    #    target_host=target_host,
    #    params=params,
    #    ops=(relay.op.get("nn.conv2d"),),
    #)
    # the last layer in resnet
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 32, 32, 512, 256, 1, 1, (1, 1), (0, 0)
    tasks = autotvm.task.create(
        "tutorial/conv2d_no_batching", args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,target_host=target_host
    )

    # run tuning tasks
    print("Tuning...")
    tune_tasks([tasks], **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file) as dispatch_context:
        best_config = dispatch_context.query(tasks.target, tasks.workload)
        print("\nBest config:")
        print(best_config)
        print("Compile...")
        with tvm.target.Target("opencl"):
            s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
            lib = tvm.build(s, arg_bufs, target_host=target_host)
            func=lib
            #print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
        # export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk

            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, "0.0.0.0", TRACKER_PORT, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
        w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
        c_np = conv2d_nchw_python(a_np, w_np, strides, padding)
        ctx = remote.context(str(target), 0)
        a_tvm = tvm.nd.array(a_np, ctx=ctx, dtype = arg_bufs[0].dtype)
        w_tvm = tvm.nd.array(w_np, ctx=ctx, dtype = arg_bufs[1].dtype)
        c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx,dtype = arg_bufs[2].dtype)
        time_f = rlib.time_evaluator(rlib.entry_name, ctx, number=10)
        cost = time_f(a_tvm, w_tvm, c_tvm).mean
        print("Time cost of this operator: %f" % cost)
        
        tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
        #ctx = remote.context(str(target), 0)
        #module = runtime.GraphModule(rlib["default"](ctx))
        #data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        #module.set_input("data", data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        #ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=30)
        #prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        #print(
        #    "Mean inference time (std dev): %.2f ms (%.2f ms)"
        #    % (np.mean(prof_res), np.std(prof_res))
        #)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)



