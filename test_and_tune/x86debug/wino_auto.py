import logging
import sys
import numpy as np

import tvm
from tvm import te
import tvm.topi
import tvm.topi.testing
from tvm import topi
from tvm.topi.testing import conv2d_nchw_python

from tvm import autotvm

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
data = te.placeholder((N, CI, H, W), name='data')
kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
#conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype='float32')
#cfg = autotvm.get_config()
task = autotvm.task.create("conv2d_nchw_winograd.cuda",
                           args=(data, kernel, strides, padding, 1, 'float32'),
                           target='cuda')
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
tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(
    n_trial=20,
    measure_option=measure_option,
    callbacks=[autotvm.callback.log_to_file("conv2d.log")],
)

#inspect the best config
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
