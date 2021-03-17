import numpy as np
import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_runtime
import os

use_ndk = True
def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)

    return mod, params, input_shape, output_shape


# Define the neural network and compilation target.
network = "mobilenet"
batch_size = 1
layout = "NHWC"
# Set this to True if you use ndk tools for cross compiling
# Path to cross compiler
#os.environ["TVM_NDK_CC"] = "/usr/bin/aarch64-linux-gnu-g++"
#target_host = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu")
arch = "arm64"
target_host = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)
target = tvm.target.Target("opencl -device=mali")
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)


#################################################################
# Start an RPC Tracker and Register Devices to the Tracker
# --------------------------------------------------------
# Please refer to the "Start RPC Tracker" and "Register Devices to RPC Tracker" setions
# in this :ref:`tutorial <tutorials-autotvm-start-rpc-tracker>` to start an RPC tracker
# and register devices to the tracker.

# Replace this with the device key in your tracker
device_key = "Adreno640"


#################################################################
# Extract Search Tasks
# --------------------
# Next, we extract the search tasks and their weights from a network.
# The weight of a task is the number of appearances of the task's subgraph
# in the whole network.
# By using the weight, we can approximate the end-to-end latency of the network
# as :code:`sum(latency[t] * weight[t])`, where :code:`latency[t]` is the
# latency of a task and :code:`weight[t]` is the weight of the task.
# The task scheduler will just optimize this objective.

# Extract tasks from the network
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, target_host)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)
######################################################################
# .. note:: How to get the hardware parameters from remote device
#
#   .. code-block:: python
#
#     from tvm.auto_scheduler.utils import request_remote
#     remote = request_remote(device_key, "0.0.0.0", 9090)
#     ctx = remote.cl()
#     max_shared_memory_per_block = ctx.max_shared_memory_per_block
#     # There is no explicit local memory limition
#     # so we can use INT32_MAX to disalbe the check on local_memory.
#     max_local_memory_per_block = 2147483647 # INT32_MAX
#     max_threads_per_block = ctx.max_threads_per_block
#     max_vthread_extent = int(ctx.warp_size / 4) if int(ctx.warp_size / 4) > 1 else ctx.warp_size
#     warp_size = ctx.warp_size
#     hardware_params = auto_scheduler.HardwareParams(-1, 16, 64,
#                                                     max_shared_memory_per_block, max_local_memory_per_block,
#                                                     max_threads_per_block, max_vthread_extent, warp_size)
#
#  Now you could pass it to search task and tune
#
#   .. code-block:: python
#
#     tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target, target_host, hardware_params)
#

#################################################################
# Tuning and Evaluate
# -------------------
# Now, we set some options for tuning, launch the search tasks and evaluate the end-to-end performance
#
# * :code:`num_measure_trials` is the number of measurement trials we can use during the tuning.
#   You can set it to a small number (e.g., 200) for a fast demonstrative run.
#   In practice, we recommend setting it around :code:`800 * len(tasks)`,
#   which is typically enough for the search to converge.
#   For example, there are 29 tasks in resnet-50, so we can set it as 20000.
#   You can adjust this parameter according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a log file,
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions`,
#   :any:`auto_scheduler.LocalRunner` for more parameters.
#


def tune_and_evaluate():
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200000,  # change this to 20000 to achieve the best performance
        builder=auto_scheduler.LocalBuilder(build_func="ndk" if use_ndk else "default"),
        runner=auto_scheduler.RPCRunner(
            device_key, host="0.0.0.0", port=9090,number=10, repeat=3, timeout=50
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


# We do not run the tuning in our webpage server since server doesn't have mali gpu.
# Uncomment the following line to run it by yourself.

tune_and_evaluate()
