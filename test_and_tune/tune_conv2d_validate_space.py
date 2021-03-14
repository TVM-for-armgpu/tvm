
import logging
import sys
import numpy as np

import tvm
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import tvm.testing

from tvm import autotvm


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
    #conv_pl.dtype = "climgfloatw32"
    s = topi.mali.schedule_conv2d_NCHWc_io(conv_pl)
    #print(tvm.lower(s, [data_pl,kernel_pl,conv_pl], simple_mode=True))
    #exit(0)

    return s, [data_pl,kernel_pl,conv_pl]


######################################################################
# Step 2:  Search through the space
# ---------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBoostTuner` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this template

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 1, 1, (1, 1), (1, 1)
task = autotvm.task.create(
    "tutorial/conv2d_no_batching", args=(N, H, W, CO, CI, KH, KW, strides, padding), target="opencl"
)
print(task.config_space)

# Use local gpu, measure 10 times for every config to reduce variance
# The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
)

# Begin tuning, log records to file `conv2d.log`
# During tuning we will also try many invalid configs, so you are expected to
# see many error reports. As long as you can see non-zero GFLOPS, it is okay.
tuner = autotvm.tuner.GridSearchTuner(task)
tuner.tune(
    n_trial=100000,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("conv2d.log")],
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
    with tvm.target.Target("opencl"):
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
