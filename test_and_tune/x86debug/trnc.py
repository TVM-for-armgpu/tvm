

import numpy as np
import tvm
import os
from tvm import te,topi
H=8
W =8
H_P, W_P = H, W

PACK4=4
CI=256
CO=256
K_P=CO//PACK4
C_P=CI//PACK4
in_channel = CI
out_channel = CO
in_size=H
open_image = 1
ddtype = 'float32'
if open_image:
    ddtype = 'climgfloatr32'
data_pl = te.placeholder((1, C_P, H, W, PACK4),
                         name='data', dtype=ddtype)
kernel_pl = te.placeholder((CI, K_P, 1, 1, 1, PACK4),
                           name='filter', dtype=ddtype)
target ="opencl"
ctx = tvm.context(target, 0)
with tvm.target.Target(target):
    conv_pl = topi.mali.conv2d_NCHWc(data_pl, kernel_pl, 1, 1,
                                     0, 'NCHWc', 'NCHWc', ddtype.replace('r','w'))
    conv_pl.dtype='climgfloatw32'
    s = topi.mali.schedule_conv2d_NCHWc(conv_pl)
    #print(tvm.lower(s, [data, kernel, conv], simple_mode=True))
    lib = tvm.build(s, [data_pl, kernel_pl, conv_pl])
    func=lib
print("------opencl code------")
print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")

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

#===============tvm data format
a_tvm = tvm.nd.array(a_np_tvm, ctx=ctx, dtype=data_pl.dtype)
w_tvm = tvm.nd.array(w_np_tvm, ctx=ctx, dtype=kernel_pl.dtype)
c_tvm = tvm.nd.empty((1, C_P, H, W, PACK4), ctx=ctx, dtype=conv_pl.dtype)
#===================run tvm -===========
rlib = func
time_f = rlib.time_evaluator(rlib.entry_name, ctx, number=1)
cost = time_f(a_tvm, w_tvm, c_tvm).mean
print("Time cost of this operator: %f,%f GFLOPs" % (cost, 1 * H * W * 512 * 256 * 2 / cost / 1000000000))

#============valide answer=========
A = a_np.astype("float32")
W = w_np.astype("float32")
C = W.dot(A)
C = C.reshape(K_P*PACK4, H_P*W_P).T.reshape(K_P*H_P*W_P, 4)
B1 = C[0::K_P, :]

for i in range(K_P-1):
    B1 = np.vstack((B1, C[i+1::K_P, :]))
C = B1.reshape(K_P, H_P * W_P * PACK4)
Ctvm = c_tvm.asnumpy().reshape(K_P, H_P*W_P*PACK4)
np.testing.assert_allclose(C, Ctvm, rtol=1e-2)
print("\nanswer::correct!,Convolution: %fms, about %f GFLOPS" %
      (cost*1e3, out_channel*in_channel*in_size*in_size*2/cost/1e9))

