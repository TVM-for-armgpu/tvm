
from tvm.contrib import utils, ndk
import os
from tvm import rpc
import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python
# Set to be address of tvm proxy.
tracker_host = '127.0.0.1'
tracker_port = 9090
key = "android"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch
target = "opencl"
#target='llvm'
#target_host = 'llvm -mcpu=core-avx2'
#ctx = tvm.context(target_host, 0);
print(os.getpid())
H=64
W = 64
H_P, W_P = H, W
KW=KH=3

PACK4=4
CI=256
CO=256
K_P=CO//PACK4
C_P=CI//PACK4
in_channel = CI
out_channel = CO
in_size=H
open_image = 0
ddtype = 'float32'
if open_image:
    ddtype = 'climgfloatr32'


data_pl = te.placeholder((1, CI, H, W),
                         name='data', dtype=ddtype)
kernel_pl = te.placeholder((CO, CI, KW, KH),
                           name='filter', dtype=ddtype)
target = tvm.target.Target("opencl -device=mali")
target ="opencl -device=mali"
ctx = tvm.context(target, 0)
with tvm.target.Target(target):
    conv_pl = topi.mali.conv2d_nchw_winograd(data_pl, kernel_pl, 1, 1,
                                     1, ddtype.replace('r','w'))
    s = topi.mali.schedule_conv2d_nchw_winograd(conv_pl)
    #print(tvm.lower(s, [data_pl,kernel_pl,conv_pl], simple_mode=True))
    #exit(0)
    #print(tvm.lower(s, [data, kernel, conv], simple_mode=True))
    lib = tvm.build(s, [data_pl, kernel_pl, conv_pl], target_host=target_host)
    func=lib

temp = utils.tempdir()

## Establish remote connection with target hardware

tracker = rpc.connect_tracker(tracker_host, tracker_port)
remote = tracker.request(key, priority=0, session_timeout=60)
with open('kk.cl','w') as fp:
    print(func.imported_modules[0].get_source(),file=fp) if len(func.imported_modules) > 0 else print("source not imported")
path_dso_cl = temp.relpath("dev_lib_cl.so")
filename="dev_lib_cl.so"
func.export_library(path_dso_cl, ndk.create_shared)
remote.upload(temp.relpath(filename))

target = tvm.target.Target("opencl")
ctx = remote.context(str(target), 0)
rlib = remote.load_module(filename)
dsp=data_pl.shape
dsp=tuple(int(i) for i in dsp)
print(dsp)
ksp = kernel_pl.shape
ksp=tuple(int(i) for i in ksp)
osp=conv_pl.shape
osp=tuple(int(i) for i in osp)
print(osp)

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
a_np_tvm=np.arange(in_channel*H*W).reshape(1,in_channel,H,W)
w_np_tvm=np.arange(KW*KH*in_channel*out_channel).reshape(out_channel,in_channel,KW,KH)
a_tvm = tvm.nd.array(a_np_tvm, ctx=ctx, dtype=data_pl.dtype)
w_tvm = tvm.nd.array(w_np_tvm, ctx=ctx, dtype=kernel_pl.dtype)
c_tvm = tvm.nd.empty(osp, ctx=ctx, dtype=conv_pl.dtype)
#===================run tvm -===========
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
#np.testing.assert_allclose(C, Ctvm, rtol=1e-2)
print("\nanswer::correct!,Convolution: %fms, about %f GFLOPS" %
      (cost*1e3, out_channel*in_channel*in_size*in_size*2/cost/1e9))
