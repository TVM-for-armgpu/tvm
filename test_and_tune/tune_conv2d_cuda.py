import logging
import sys
import numpy as np

import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

from tvm import autotvm


TRACKER_PORT=9090
@autotvm.template("tutorial/conv2d_no_batching")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel_f")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    s = te.create_schedule([conv.op])

    ##### space definition begin #####
    n, f, y, x = s[conv].op.axis
    rc, ry, rx = s[conv].op.reduce_axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_f", f, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=3)
    cfg.define_split("tile_rx", rx, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # inline padding
    pad_data = s[conv].op.input_tensors[0]
    s[pad_data].compute_inline()
    data, raw_data = pad_data, data

    output = conv
    OL = s.cache_write(conv, "local")

    # create cache stage
    AA = s.cache_read(data, "shared", [OL])
    WW = s.cache_read(kernel, "shared", [OL])
    AL = s.cache_read(AA, "local", [OL])
    WL = s.cache_read(WW, "local", [OL])

    # tile and bind spatial axes
    n, f, y, x = s[output].op.axis
    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)
    kernel_scope = n  # this is the scope to attach global config inside this kernel

    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, f, y, x = s[OL].op.axis
    rc, ry, rx = s[OL].op.reduce_axis
    rco, rcm, rci = cfg["tile_rc"].apply(s, OL, rc)
    ryo, rym, ryi = cfg["tile_rx"].apply(s, OL, ry)
    rxo, rxm, rxi = cfg["tile_ry"].apply(s, OL, rx)
    s[OL].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, n, f, y, x)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    s[AL].compute_at(s[OL], rxm)
    s[WL].compute_at(s[OL], rxm)

    # cooperative fetching
    for load in [AA, WW]:
        n, f, y, x = s[load].op.axis
        fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_f"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    # tune unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    return s, [raw_data, kernel, conv]


######################################################################
# Step 2:  Search through the space
# ---------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBoostTuner` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this template

# Use local gpu, measure 10 times for every config to reduce variance
# The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
target = tvm.target.Target("opencl -device=mali")

# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target host is used for cross compilation. You can query it by :code:`gcc -v` on your device.
#target_host = "llvm -mtriple=aarch64-linux-gnu"
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch

# logging config (for printing tuning log to screen)
#logging.getLogger("autotvm").setLevel(logging.DEBUG)
#logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 1, 1, (1, 1), (1, 1)
task = autotvm.task.create(
    "tutorial/conv2d_no_batching", args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,target_host=target_host
)
print(task.config_space)


# Also replace this with the device key in your tracker
device_key = "MaliG72"

# Set this to True if you use android phone
use_android = True

#### TUNING OPTION ####
network = "conv2d_cuda_72"
log_file = "%s.%s.log" % (device_key, network)
dtype = "float32"
measure_option= autotvm.measure_option(
    builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
    runner=autotvm.RPCRunner(
        device_key,
        host="0.0.0.0",
        port=TRACKER_PORT,
        number=10,
        timeout=15,
    ),
)

# Begin tuning, log records to file `conv2d.log`
# During tuning we will also try many invalid configs, so you are expected to
# see many error reports. As long as you can see non-zero GFLOPS, it is okay.
tsk_trial = 20000
prefix = "[Task %2d/%2d] " % (0 + 1, 1)
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=tsk_trial,
    measure_option=measure_option,
    callbacks=[
        autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
        autotvm.callback.log_to_file(log_file)
    ],
)

#########################################################################
# Finally we can inspect the best config from log file, check correctness,
# and measure running time.

# inspect the best config
dispatch_context = autotvm.apply_history_best("conv2d.log")
best_config = dispatch_context.query(task.target, task.workload)
print("\nBest config:")
print(best_config)

# apply history best from log file
with autotvm.apply_history_best("conv2d.log"):
    with tvm.target.Target("cuda"):
        s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
        func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

ctx = tvm.gpu()
a_tvm = tvm.nd.array(a_np, ctx=ctx)
w_tvm = tvm.nd.array(w_np, ctx=ctx)
c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
func(a_tvm, w_tvm, c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

# Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
# and the overhead of kernel launch. You can also use nvprof to validate the result.
evaluator = func.time_evaluator(func.entry_name, ctx, number=400)
print("Time cost of this operator: %f" % evaluator(a_tvm, w_tvm, c_tvm).mean)