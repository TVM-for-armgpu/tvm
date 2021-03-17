from tvm.contrib import ndk
from tvm.contrib.utils import tempdir
import os
from tvm import relay, autotvm

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
import sys
import logging

logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


######################################################################
# Create the search task
# ^^^^^^^^^^^^^^^^^^^^^^
# We then create a search task for the last convolution layer in the resnet.

target = tvm.target.Target("opencl -device=mali")
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch

# Use the last layer in ResNet-50
N, H, W, CO, CI, KH, KW, strides, padding = 1, 40, 40, 32, 32, 1, 1, (1, 1), (0, 0)
task = auto_scheduler.SearchTask(
func=conv2d_layer, args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,target_host=target_host
)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

######################################################################
# Next, we set parameters for the auto-scheduler. These parameters
# mainly specify how we do the measurement during the search.
#
# * :code:`measure_ctx` launches a different process for measurement to
#   provide isolation. It can protect the master process from GPU crashes
#   during measurement and avoid other runtime conflicts.
# * :code:`min_repeat_ms` defines the minimum duration of one "repeat" in every measurement.
#   This can warmup the GPU, which is necessary to get accurate measurement results.
#   Typically, we recommend a value >= 300 ms.
# * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
#   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
#   good value for the search to converge. You can do more trials according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a file `conv2d.json`.
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions`,
#   :any:`auto_scheduler.LocalRPCMeasureContext` for more parameters.

log_file = "conv2d.json"
device_key = "Adreno640"
use_android = True
use_ndk=True
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=2000,  # change this to 20000 to achieve the best performance
    builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
    runner=auto_scheduler.RPCRunner(
        device_key, host="0.0.0.0", port=9090, number=10,timeout=50
    ),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)
#tune_option = auto_scheduler.TuningOptions(
#    num_measure_trials=2000,  # change this to 1000 to achieve the best performance
#    builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_android else "default"),
#    runner=auto_scheduler.RPCRunner(
#        device_key,
#        host="0.0.0.0",
#        port=9090,
#        number=10,
#        timeout=5,
#    ),
#    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
#    verbose=2,
#)

######################################################################
# Run the search
# ^^^^^^^^^^^^^^
# Now we get all inputs ready. Pretty simple, isn't it?
# We can kick off the search and let the auto-scheduler do its magic.
# After some measurement trials, we can load the best schedule from the log
# file and apply it.

def tune_and_evaluate(tasks, task_weights):
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20000,  # change this to 20000 to achieve the best performance
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
        runner=auto_scheduler.RPCRunner(
            device_key, host="0.0.0.0", port=9090, repeat=3, timeout=50
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

    # Compile the whole network
    print("Compile...")
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(
            opt_level=3, config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)

    # Create graph runtime
    print("=============== Request Remote ===============")
    from tvm.auto_scheduler.utils import request_remote

    remote = request_remote(device_key, "0.0.0.0", 9090)
    ctx = remote.cl()
    from tvm.contrib import utils, ndk

    temp = utils.tempdir()
    filename = "deploy_lib.so"
    path_lib = temp.relpath(filename)
    lib.export_library(path_lib, ndk.create_shared)
    remote.upload(path_lib)
    loaded_lib = remote.load_module(filename)
    module = graph_runtime.GraphModule(loaded_lib["default"](ctx))
    data = (np.random.uniform(size=input_shape)).astype(dtype)
    data_tvm = tvm.nd.array(data)
    module.set_input("data", data_tvm)

    # Evaluate
    print("Evaluate inference time cost...")
    ftimer = module.module.time_evaluator("run", ctx, repeat=3, min_repeat_ms=500)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
    )
# Run auto-tuning (search)
#task.tune(tune_option)
#tune_and_evaluate([task],[1])
# Apply the best schedule
sch, args = task.apply_best(log_file)

######################################################################
# We can lower the schedule to see the IR after auto-scheduling.
# The auto-scheduler correctly performs optimizations including multi-level tiling,
# cooperative fetching, unrolling and operator fusion.

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

######################################################################
# Check correctness and evaluate performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We build the binary and check its correctness and performance.


# Check correctness
data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)

print("Equivalent python schedule:")
print(task.print_best(log_file, print_mode="schedule"))

print("CUDA source code:")
TRACKER_PORT=9090
print(task.print_best(log_file, print_mode="opencl"))
    # compile kernels with history best records
with autotvm.apply_history_best(log_file) as dispatch_context:
    best_config = dispatch_context.query(tasks.target, tasks.workload)
    print("\nBest config:")
    print(best_config)
    print("Compile...")
    with tvm.target.Target("opencl -device=mali"):
        lib = tvm.build(sch, args, target_host=target_host)
    # export library
    tmp = tempdir()
    if use_android:
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
    #bias_np = np.random.uniform(size=(1, CO, 1, 1)).astype(np.float32)
    c_np = conv2d_nchw_python(a_np, w_np, strides, padding)
    out_np=c_np
    #out_np = np.maximum(c_np + bias_np, 0.0)
    ctx = remote.context(str(target), 0)
    a_tvm = tvm.nd.array(a_np, ctx=ctx, opencl_image=open_image)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
    #bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
    time_f = rlib.time_evaluator(rlib.entry_name, ctx, number=10)
    cost = time_f(a_tvm, w_tvm, c_tvm).mean
    print("Time cost of this operator: %f" % cost)
    
    tvm.testing.assert_allclose(out_np, c_tvm.asnumpy(), rtol=1e-2)
