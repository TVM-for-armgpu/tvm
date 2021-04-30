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

import time
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
#logging.getLogger("autotvm").setLevel(logging.DEBUG)
#logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.

TRACKER_PORT=9090
@autotvm.template("conv2d_no_batching")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    H_P, W_P = H, W
    PACK4=4
    K_P=CO//PACK4
    C_P=(CI+3)//PACK4
    in_channel = CI
    out_channel = CO
    in_size=H
    open_image = 1
    ddtype = 'float32'
    if open_image:
        ddtype = 'climgfloatr32'
    data_pl = te.placeholder((1, C_P, H, W, PACK4),
                             name='data', dtype=ddtype)
    kernel_pl = te.placeholder((CI,K_P, KH, KW, 1, PACK4),
                               name='filter', dtype=ddtype)
    if "Mali" in device_key:
        conv_pl = topi.mali.conv2d_NCHWc_mali_io(data_pl, kernel_pl, stride, padding, 1,
                                            'NCHWc', 'NCHWc',
                                            ddtype.replace('r', 'w'))
        s = topi.mali.schedule_conv2d_NCHWc_mali_io(conv_pl)
    else:
        conv_pl = topi.mali.conv2d_NCHWc_io(data_pl, kernel_pl, stride, padding, 1,
                                            'NCHWc', 'NCHWc',
                                            ddtype.replace('r', 'w'))
        s = topi.mali.schedule_conv2d_NCHWc_io(conv_pl)
    #print(tvm.lower(s, [data_pl,kernel_pl,conv_pl], simple_mode=True))
    #lib = tvm.build(s, [data_pl,kernel_pl,conv_pl])
    #print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
    #exit(0)

    return s, [data_pl,kernel_pl,conv_pl]

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

# Also replace this with the device key in your tracker
device_key = "Adreno640"

# Set this to True if you use android phone
use_android = True

ARCH_DETAIL = {
    "max_local_memory_per_thread": int(181 * 4),
    "max_shared_memory_per_block": 32768,
    "min_threads_per_block": int(64 * 0.8),
    "max_threads_per_block": 1024,
    "max_thread_x": 1024,
    "max_thread_y": 1024,
    "max_thread_z": 1024,
    "max_vthread": 1,
    "max_vector_bytes": 16,
    "max_vector_elems": 4,
    "buf_top_cache_bytes": 65536,
    "img_top_cache_bytes": 1024,
    "img_alloc_dtype_bytes": 2,
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
    use_transfer_learning=False
    # create tmp log file
    tmp_log_file = os.path.basename(log_filename) + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        if in_check_point(tsk.args):
            print(f"skip finished {i}th task",tsk.args)
            continue
        print(f"start tuning {i+1}th task",tsk.args)
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
        time.sleep(60*1)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.
convshape = [
    [1, 0, 28, 28, 256, 256, 1, 1, [1, 1], [0, 0, 0, 0], [0, 0]],
]


def in_check_point(shape:str) -> bool:
    #return False
    import json
    if isinstance(shape, tuple):
        shape = list(shape)

    if len(shape) == 11:
        shape.pop(1)
    #shape.pop(-1)
    for ind, sp in enumerate(shape):
        if isinstance(sp, tuple):
            shape[ind] = list(sp)
    if not os.path.exists(log_file):
        return False
    with open(log_file) as fp:
        for sp_hist in fp:
            sp_hist_json = json.loads(sp_hist)
            if sp_hist_json["input"][2] == shape:
                return True
    return False


def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...",os.getpid())
    #mod, params, input_shape, _ = get_network(network, batch_size=1)
    #tasks = autotvm.task.extract_from_program(
    #    mod["main"],
    #    target=target,
    #    target_host=target_host,
    #    params=params,
    #    ops=(relay.op.get("nn.conv2d"),),
    #)
    # the last layer in resnet
    sys.path.append('remote_tmali')
    sys.path.append('./')
    tasks = []
    for ith, shape in enumerate(convshape):
        N, _, H, W, CI, CO, KH, KW, strides, padding, dialte = shape
        if KH < -5  or KW < -5:
            if KH < -5:
                H=(1+(H+padding[0]+padding[2])//strides[0])//2*2*strides[0]-padding[0]-padding[2]
            if KW < -5:
                W=(1+(W+padding[1]+padding[3])//strides[1])//2*2*strides[1]-padding[1]-padding[3]
        else:
            H=(1+H)//2*2
            W=(1+W)//2*2
        task = autotvm.task.create(
            "conv2d_no_batching", args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target
        )
        tasks.append(task)

        # run tuning tasks
    print(f"Tuning...{ith}th task", shape, device_key)
    tasks.reverse()
    tune_tasks(tasks, **tuning_opt)


    # compile kernels with history best records
    #with autotvm.apply_history_best(log_file) as dispatch_context:
    #    best_config = dispatch_context.query(tasks[0].target, tasks[0].workload)
    #    print("\nBest config:")
    #    print(best_config)
    #    print("Compile...")
    if 1:
        with tvm.target.Target("opencl"):
            s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
            lib = tvm.build(s, arg_bufs)
            func=lib
            print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
        # export library
        tmp = tempdir()
        filename = "net.stackvm"
        lib.export_library(tmp.relpath(filename))

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, "127.0.0.1", TRACKER_PORT, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        a_s=tuple(int(i) for i in arg_bufs[0].shape)
        w_s=tuple(int(i) for i in arg_bufs[1].shape)
        #a_np = np.random.uniform(size=a_s).astype(np.float32)
        #w_np = np.random.uniform(size=w_s).astype(np.float32)
        # Algorithm
        PACK4 = 4
        W_P = W
        H_P = H
        C_P = CI//PACK4
        K_P=CO//PACK4
        in_channel = CI
        out_channel = CO
        in_size=H
        ctx = remote.context(str(target), 0)

        #================test data=================
        ao_np = np.arange(in_channel*in_size*in_size)
        a_np = ao_np
        a_np = a_np.reshape(C_P*PACK4, H_P*W_P)
        wo_np = np.arange(in_channel*out_channel*KH*KW).reshape(out_channel,in_channel)
        w_np = wo_np
        #==A-tvm data prepare
        a_np_tvm = a_np.T.reshape(C_P*H_P*W_P, 4)
        B1 = a_np_tvm[0::C_P, :]
        for i in range(C_P-1):
            B1 = np.vstack((B1, a_np_tvm[i+1::C_P, :]))
        a_np_tvm = B1.reshape(1, C_P, H_P, W_P, PACK4) * 1.0
        #==W-tvm data prepare
        w_np_tvm = w_np.T.reshape(w_s)*1.0
        #============valide answer=========
        Anp = a_np.astype("float32")
        Wnp = w_np.astype("float32")
        Cnp = Wnp.dot(Anp)
        Cnp = Cnp.reshape(K_P*PACK4, H_P*W_P).T.reshape(K_P*H_P*W_P, 4)
        B1 = Cnp[0::K_P, :]
        for i in range(K_P-1):
            B1 = np.vstack((B1, Cnp[i+1::K_P, :]))
        c_np = B1.reshape(K_P, H_P * W_P * PACK4)

        #===============tvm data format
        a_tvm = tvm.nd.array(a_np_tvm, ctx=ctx, dtype=arg_bufs[0].dtype)
        w_tvm = tvm.nd.array(w_np_tvm, ctx=ctx, dtype=arg_bufs[1].dtype)
        c_tvm = tvm.nd.empty(arg_bufs[2].shape, ctx=ctx, dtype=arg_bufs[2].dtype)
        ####numpy calculate the answer over

        #c_np = conv2d_nchw_python(a_np, w_np, strides, padding)
        time_f = rlib.time_evaluator(rlib.entry_name, ctx, number=10)
        cost = time_f(a_tvm, w_tvm, c_tvm).mean
        GFLOPS = W_P*H_P*CI*CO*2/cost/1e9
        print("Time cost of this operator: %f, %f gflops" %( cost,GFLOPS))

        c_tvm_o = c_tvm.asnumpy().reshape(K_P,H_P*W_P*PACK4)
        tvm.testing.assert_allclose(c_np, c_tvm_o, rtol=1e-2)

        #ctx = remote.context(str(target), 0)
        #module = runtime.GraphModule(rlib["default"](ctx))
        #data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        #module.set_input("data", data_tvm)

        # evaluate
        print("answer check passed. Evaluate inference time cost... ")
        ftimer = rlib.time_evaluator(rlib.entry_name, ctx, number=10, repeat=30)
        a=ftimer(a_tvm, w_tvm, c_tvm).results
        prof_res = np.array(a) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms) %f GFLOPS"
            % (np.mean(prof_res), np.std(prof_res),W*H*CI*CO*2/np.mean(prof_res)/1e6)
        )


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("please give a device_key")
        exit(0)

    device_key = sys.argv[1]
    assert device_key in ["MaliG72", "MaliG76", "Adreno640", "Adreno630"]

    #### TUNING OPTION ####
    network = "topinhw3c1_packed"
    log_file = "%s.%s.log" % (device_key, network)
    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 1000,
        "use_transfer_learning": False,
        "early_stopping": 1000,
        "measure_option":
        autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func="stackvm"),
            runner=autotvm.RPCRunner(
                device_key,
                host="127.0.0.1",
                port=TRACKER_PORT,
                number=10,
                timeout=5,
                arch_detail=ARCH_DETAIL,
            ),
        ),
    }
    tune_and_evaluate(tuning_option)
