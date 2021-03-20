import tvm
from tvm import relay, autotvm
from tvm.contrib.utils import tempdir
from tvm.contrib import ndk
from tvm import te, topi
from tvm.topi import nn
from numbers import Integral
import numpy as np
def get_const_tuple(in_tuple):
    """Verifies input tuple is IntImm or Var, returns tuple of int or Var.

    Parameters
    ----------
    in_tuple : tuple of Expr
        The input.

    Returns
    -------
    out_tuple : tuple of int
        The output.
    """
    ret = []
    ana = None
    for elem in in_tuple:
        if isinstance(elem, (tvm.tir.Var, tvm.tir.expr.Any)):
            ret.append(elem)
        elif not isinstance(elem, (tvm.tir.IntImm, int)):
            ana = tvm.arith.Analyzer() if ana is None else ana
            elem = ana.simplify(elem)
            if not isinstance(elem, tvm.tir.IntImm):
                ret.append(elem)
            else:
                ret.append(get_const_int(elem))
        else:
            ret.append(get_const_int(elem))
    return tuple(ret)
def get_const_int(expr):
    """Verifies expr is integer and get the constant value.

    Parameters
    ----------
    expr : tvm.Expr or int
        The input expression.

    Returns
    -------
    out_value : int
        The output.
    """
    if isinstance(expr, Integral):
        return expr
    if not isinstance(expr, tvm.tir.IntImm):
        ana = tvm.arith.Analyzer()
        expr = ana.simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        raise ValueError("Expect value to be constant int")
    return int(expr.value)


H=40
W = 40
H_P, W_P = H, W
KW=KH=3
PACK4=4
CI=16
CO=16
K_P=CO//PACK4
C_P=CI//PACK4
in_channel = CI
out_channel = CO
in_size=H
open_image = 0
ddtype = 'float32'

data_pl = te.placeholder((1, CI, H, W),
                         name='data', dtype=ddtype)
kernel_pl = te.placeholder((CO, CI, KW, KH),
                           name='filter', dtype=ddtype)



data, kernel=data_pl,kernel_pl
N, CI, IH, IW = get_const_tuple(data.shape)
dilation=1
strides=1
padding=1
tile_size=6
out_dtype='float32'
if isinstance(dilation, int):
    dilation_h = dilation_w = dilation
else:
    dilation_h, dilation_w = dilation

if len(kernel.shape) == 4:
    if dilation_h != 1 or dilation_w != 1:
        kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))
    pre_computed = False
    CO, _, KH, KW = get_const_tuple(kernel.shape)
else:
    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
    pre_computed = True
    H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
    CO *= VC
    KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))

assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1
data_pad = nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")

r = KW
m = tile_size
alpha = m + r - 1

A, B, G = nn.winograd_util.winograd_transform_matrices(m, r, out_dtype)

H = (IH + pt + pb - 3) // HSTR + 1
W = (IW + pl + pr - 3) // WSTR + 1
nH, nW = (H + m - 1) // m, (W + m - 1) // m
P = N * nH * nW
bna=nH
bnb=nW
P_round = (P + bnb - 1) // bnb * bnb

input_tile = te.compute(
    (CI, P_round // bnb, alpha, alpha, bnb),
    lambda ci, b, eps, nu, bb: tvm.tir.if_then_else(
        b * bnb + bb < P,
        data_pad[(b * bnb + bb) // (nH * nW)][ci][(b * bnb + bb) // nW % nH * m + eps][
            (b * bnb + bb) % nW * m + nu],
        tvm.tir.const(0, data_pad.dtype),
    ),
    name="d",
)

# transform image
r_a = te.reduce_axis((0, alpha), "r_a")
r_b = te.reduce_axis((0, alpha), "r_b")
V = te.compute(
    (alpha, alpha, P_round // bnb, CI, bnb),
    lambda eps, nu, p, ci, vp: te.sum(
        input_tile[ci][p][r_a][r_b][vp] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
    ),
    name="V",
)

BOUT=V
s = te.create_schedule(BOUT.op)
s[input_tile].compute_inline()
dataL = s.cache_read(data_pad, "local", [input_tile])
cfg = autotvm.get_config()
cfg.add_flop(2.0*np.prod([N, H, W, CO, CI, 8]))
sc,se,sh,sw,sb  = s[input_tile].op.axis
sco,sci=s[input_tile].split(sc,factor=4)


s[input_tile].bind(sci, te.thread_axis("blockIdx.z"))
s[input_tile].bind(sco, te.thread_axis("blockIdx.y"))
s[input_tile].bind(se, te.thread_axis("blockIdx.x"))
s[input_tile].bind(sb, te.thread_axis("threadIdx.z"))
s[input_tile].bind(sw, te.thread_axis("threadIdx.y"))
s[input_tile].bind(sh, te.thread_axis("threadIdx.x"))

se,su,sp,sc,sv  = s[V].op.axis

sco,sci=s[V].split(sc,factor=4)
s[V].bind(se, te.thread_axis("blockIdx.z"))
s[V].bind(sco, te.thread_axis("blockIdx.y"))
s[V].bind(su, te.thread_axis("blockIdx.x"))
s[V].bind(sp, te.thread_axis("threadIdx.z"))
s[V].bind(sci, te.thread_axis("threadIdx.y"))
s[V].bind(sv, te.thread_axis("threadIdx.x"))
s[dataL].compute_at(s[V],r_a)


target = tvm.target.Target("opencl")
#target = tvm.target.Target("opencl -device=mali")

# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target host is used for cross compilation. You can query it by :code:`gcc -v` on your device.
#target_host = "llvm -mtriple=aarch64-linux-gnu"
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch

# Also replace this with the device key in your tracker
device_key = "Adreno640"

# Set this to True if you use android phone
use_android = True
#print(tvm.lower(s,[data,BOUT], simple_mode=True))
with tvm.target.Target("opencl"):
    func=tvm.build(s,[data,BOUT],target_host=target_host)
print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")

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
a_np_tvm=np.arange(in_channel*H*W).reshape(1,in_channel,H,W)
w_np_tvm=np.arange(KW*KH*in_channel*out_channel).reshape(out_channel,in_channel,KW,KH)
ctx = remote.context(str(target), 0)
a_tvm = tvm.nd.array(a_np_tvm, ctx=ctx, dtype=data_pl.dtype)
w_tvm = tvm.nd.array(w_np_tvm, ctx=ctx, dtype=kernel_pl.dtype)
c_tvm = tvm.nd.empty(osp, ctx=ctx, dtype=input_tile.dtype)
func(a_tvm,c_tvm)
print(c_tvm.shape)

