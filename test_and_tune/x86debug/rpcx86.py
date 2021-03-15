import os

print(os.getpid())
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
@autotvm.template("tutorial/conv2d_no_batching")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    H_P, W_P = H, W
    PACK4=4
    K_P=CO//PACK4
    C_P=(CI+3)//PACK4
    in_channel = CI
    out_channel = CO
    in_size=H
    open_image = 0
    ddtype = 'float32'
    if open_image:
        ddtype = 'climgfloatr32'
    data_pl = te.placeholder((1, C_P, H, W, PACK4),
                             name='data', dtype=ddtype)
    kernel_pl = te.placeholder((CI,K_P, 1, 1, 1, PACK4),
                               name='filter', dtype=ddtype)
    conv_pl = topi.mali.conv2d_NCHWc_io(data_pl, kernel_pl, 1, 1,
                                     0, 'NCHWc', 'NCHWc', ddtype.replace('r','w'))
    conv_pl.dtype = "climgfloatw32"
    s = topi.mali.schedule_conv2d_NCHWc_io(conv_pl)
    #print(tvm.lower(s, [data_pl,kernel_pl,conv_pl], simple_mode=True))
    #exit(0)

    return s, [data_pl,kernel_pl,conv_pl]


#################################################################
# Replace "armv8a-linux-gnueabihf" with the correct target of your board.
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
target = tvm.target.Target('llvm  -device=arm_cpu -mtriple=arm64-linux-android')
target_host=None
arch = "arm64"
#target_host = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)
target = tvm.target.Target("opencl -device=mali")

# Also replace this with the device key in your tracker
device_key = "x86"


#### TUNING OPTION ####
dtype = 'float32'
dtype = 'climgfloatr32'

network = 'mobilenet_v1_1.0_224'
log_file = "%s.%s.log" % (dtype, network)


use_android=False
tuning_option = {
    'log_filename': log_file,
    'tuner': 'xgb',
    'n_trial': 32,
    'early_stopping': 200,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func='ndk' if use_android else 'default'),
        runner=autotvm.RPCRunner(
            device_key, host='127.0.0.1', port=9090,
            number=5,
            timeout=10,
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
#   If your model has depthwise convolution, you could consider setting
#   :code:`try_spatial_pack_depthwise` be :code:`True`, which perform better than default
#   optimization in general. For example, on ARM CPU A53 2.0GHz, we find it could boost 1.6x
#   performance of depthwise convolution on Mobilenet V1 model.

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)
    #use_transfer_learning=False

    for i, tsk in enumerate(reversed(tasks)):
        print("tuning ", tsk)
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])
        #break


    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    #os.remove(tmp_log_file)

# At commit baff99c83f9f691174434e7c78a4fee48b558547, ARM NHWC schedule is not high performance. So,
# we first switch to NCHW. Further, Relay build calls AlterOpLayout to optimize the data layout to
# NCHWc. If you want to use NHWC layout, please set use_nchw to False.
use_nchw = True

# TVM has many conv2d schedules for different platforms. As of commit
# baff99c83f9f691174434e7c78a4fee48b558547, we observed that x86 NCHWc schdules are faster than ARM
# NCHW or ARM NHWC schedule. If you want to use ARM NCHW spatial pack schedule, set this to false.
use_x86_schedules = True
if use_x86_schedules:
    # We must convert to NCHW first to use x86 schedules
    assert use_nchw
def remove_template(tasks, template_names):
    """ Removes the tasks that have a template name present in the template_names list.
    Parameters
    ----------
    tasks: list of tasks.
        The list of autotvm tasks.
    template_names : list of str.
        The list of template names that should be removed from tasks.
    Returns
    -------
    out_tasks: list of tasks.
        Filtered out tasks.
    """

    out_tasks = []
    for task in tasks:
        template_name = task.workload[0]
        if template_name not in template_names:
            out_tasks.append(task)
    return out_tasks


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.
def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    N, H, W, CO, CI, KH, KW, strides, padding = 1, 40, 40, 32, 4, 1, 1, (
	    1, 1), (0, 0)
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
            print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
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
        wo_np = np.arange(in_channel*out_channel).reshape(out_channel,in_channel)
        w_np = wo_np
        #==A-tvm data prepare
        a_np_tvm = a_np.T.reshape(C_P*H_P*W_P, 4)
        B1 = a_np_tvm[0::C_P, :]
        for i in range(C_P-1):
            B1 = np.vstack((B1, a_np_tvm[i+1::C_P, :]))
        a_np_tvm = B1.reshape(1, C_P, H_P, W_P, PACK4) * 1.0
        #==W-tvm data prepare
        w_np_tvm = w_np.T.reshape(CI, K_P, 1, 1, 1, PACK4)*1.0
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
    tune_and_evaluate(tuning_option)

