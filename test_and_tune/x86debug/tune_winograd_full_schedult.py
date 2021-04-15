import tvm
from tvm import te, topi,autotvm
from tvm.topi import nn
import tvm.testing
from tvm import te
import numpy
import numpy as np
import timeit
from tvm.topi.mali import mali_winograd as tune_winograd
from tvm.topi.testing import conv2d_nchw_python
from tvm.contrib.utils import tempdir

# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.
MD = 4
K = 3
N = 3
IC=160
OC=320
# The default tensor type in tvm
ddtype = "climgfloatr32"
out_dtype = "climgfloatw32"
#ddtype=out_dtype='float32'
# using Intel AVX2(Advanced Vector Extensions) ISA for SIMD
# To get the best performance, please change the following line
# to llvm -mcpu=core-avx2, or specific type of CPU you use
target = "opencl"
ctx = tvm.context(target, 0)

# Random generated tensor for testing

MD=4
K=3
r = K
m = 2
A, B, G = tvm.topi.nn.winograd_util.winograd_transform_matrices(m, r, 'float32')


data_shape = (1, 128 // 4, 28, 28, 4)
kernel_shape = (128, 64 // 4, 3, 3, 1, 4)
out_shape = (1, 128 // 4, 28, 28, 4)

tile_size = 2
wino_size = tile_size + 2
wino_all_size = wino_size * wino_size
winop2_tile_size = wino_all_size
alpha = wino_size
img_H = data_shape[2]
img_W = data_shape[3]
wino2H = (img_H+tile_size-1)//tile_size
wino2W = (img_W + tile_size - 1) // tile_size
oc_chunk = out_shape[1]
CO = oc_chunk*4
CI=kernel_shape[0]
NTILE = wino2H*wino2H

data_pl = te.placeholder((1, CI//4, img_H, img_W, 4),
                            name='placeholder', dtype=ddtype)
kernel_pl = te.placeholder((CI, CO//4, 3, 3,1,4),
                            name='placeholder', dtype=ddtype)
conv_pl = topi.mali.conv2d_nchw_winograd_NCHWc_io(data_pl, kernel_pl, 1, 1,
                                     1, "","",ddtype.replace('r','w'))

# Algorithm
# kernel transform
#k1 = te.reduce_axis((0, K), "k1")
#k2 = te.reduce_axis((0, K), "k2")
##G = te.placeholder((MD, K), name="G")
#filter_n = te.placeholder((IC,OC,K, K,1,4), name="filter")
#U = te.compute((IC, OC//4, wino_size, wino_size, 1, 4), lambda ic, oc, x, y, bi, bo:
#               te.sum(G[x, k2] * filter_n[ic, oc, k2, k1, bi, bo]*G[y, k1],
#                      axis=[k2, k1]),
#               name="U")
#U = tune_winograd.kernel_transform(kernel_shape, wino_size, G=G)
# Default schedule
#s = te.create_schedule(U.op)


# data transform

#k1 = te.reduce_axis((0, alpha), "r_a")
#k2 = te.reduce_axis((0, alpha), "r_b")
##B = te.placeholder((MD, MD), name="B")
#Data = te.placeholder((1,IC//4,img_H, img_W,4), name="Data")
#N=1
#pt = pl = pb = pr = tile_size - 1
#Data_pad = nn.pad(Data, (0, 0, pt, pl, 0), (0, 0, pb, pr, 0), name="Data_pad")
#def _transform_V(ic, oc, x, y, bo):
#    tx = y // wino2W * wino_size + x // wino_size
#    ty = (y % wino2W) * wino_size + x % wino_size
#    ox = tx // wino_size * tile_size + k2 - 1
#    oy = ty // wino_size * tile_size + k1 - 1
#    return te.sum(B[k2, tx % wino_size]*Data_pad[ic, oc, ox, oy, bo]*B[k1, ty % wino_size],
#                  axis=[k2, k1],
#                  )
#
#
#V = te.compute((N, IC // 4, wino_size*wino_size, ((img_H + tile_size-1) // tile_size) * ((img_W + tile_size-1) // tile_size), 4),
#               _transform_V,
#               name="V")
#V = tune_winograd.data_transform(data_shape, wino_size, B=B)

#s = te.create_schedule(V.op)

#func = tvm.build(s, [Data,V], target=target)
#assert func
#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
#print(tvm.lower(s, [Data,V], simple_mode=True))

#rc = te.reduce_axis((0, IC), "r_c")
#M = te.compute((N, OC//4, wino_all_size, wino2H * wino2W, 4), lambda n, oc, h, w, bo:
#               te.sum(U[rc, oc, h//wino_size, h % wino_size, 0, bo]*V[n, rc//4, h, w, rc % 4],
#                      axis=[rc]),
#               name="M"
#               )
#M = tune_winograd.batch_gemm(U, V)

#s = te.create_schedule(M.op)

#func = tvm.build(s, [U,V,M], target=target)
#assert func
#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
#print(tvm.lower(s, [U,V,M], simple_mode=True))
#print(tvm.lower(s, [filter_n,Data,M], simple_mode=True))



# inverse transform
#k1 = te.reduce_axis((0, alpha), "r_a")
#k2 = te.reduce_axis((0, alpha), "r_b")
#def _transform_inverse(n, oc, h, w, bo):
#    th = h // tile_size * wino_size + k2
#    tw = w // tile_size * wino_size + k1
#
#    oh = (th % wino_size + tw // wino_size) * wino_size + tw % wino_size
#    ow = tw // wino_size + (th // wino_size) * wino2W
#
#    #AT(th % tile_size, k2) * M[ox][oy] * A(k1, tw % 2)
#    return te.sum(A[k2, th % tile_size]*M[n, oc, oh, ow, bo]*A[k1, tw % tile_size],
#                  axis=[k1, k2])
#
##int th = int(h / tile_size) * wino_size + k2;
##int tw = int(w / tile_size) * wino_size + k1;
##
##int ox = (th % wino_size) * wino_size + tw % wino_size;
##int oy = int(tw / wino_size) + int(th / wino_size) * trans_image_size_block_W;
##output[h][w] += AT(th % tile_size, k2) * M[ox][oy] * A(k1, tw % 2);
#
#
#output = te.compute((1, OC//4, img_H, img_W, 4), _transform_inverse,
#                    name="output"
#                    )
#output = tune_winograd.inverse_transform(out_shape, A=A, M=M)
output = conv_pl

print(output.shape)
s = te.create_schedule(output.op)




# schedule for U
#========shedule begin=================================


def s1(s, G, filter_n, U):
    s[G].compute_inline()
    WL = s.cache_read(filter_n, "local", [U])
    UL = s.cache_write(U, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis( "threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    cp, kp, Uhp, Uwp, _, Up4  = s[U].op.axis
    cpo,cpi=s[U].split(cp,factor=4)
    kpo,kpi=s[U].split(kp,factor=4)
    s[U].bind(cpo, block_x)
    s[U].bind(kpo, block_y)
    s[U].bind(kpi, thread_x)
    s[U].bind(cpi, thread_y)
    s[U].reorder(cpo,cpi,kpi,Uhp,Uwp,Up4,kpo)

    # Schedule BL local write
    s[UL].compute_at(s[U],kpo)
    _,_,hp, wp, _, p4  = s[UL].op.axis
    s[UL].reorder(hp,wp,p4)
    s[UL].vectorize(p4)
    k1,k2 = s[UL].op.reduce_axis
    #s[UL].unroll(wp)
    #s[UL].unroll(hp)
    #s[UL].unroll(k1)
    #s[UL].unroll(k2)

    s[WL].compute_at(s[UL], hp)
    _,_,hp, wp,_, p4 = s[WL].op.axis
    s[WL].vectorize(p4)  # vectorize memory load
    #s[WL].unroll(wp)
    #s[WL].unroll(hp)
    s[U].vectorize(Up4)  # vectorize memory load
    #s[U].unroll(Uwp)  # vectorize memory load
    #s[U].unroll(Uhp)  # vectorize memory load
##========shedule end=================================
## shedule for V
##========shedule begin=================================
def s2(s,B,DataPad,V):
    s[B].compute_inline()
    _, _Data = s[V].op.input_tensors
    s[DataPad].compute_inline()
    WL = s.cache_read(DataPad, "local", [V])
    VL = s.cache_write(V, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis( "threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    n, cp, hp, wp, Vp4  = s[V].op.axis
    cpo,cpi=s[V].split(cp,factor=4)
    wp,Vwp=s[V].split(wp,factor=4)
    wpo,wpi=s[V].split(wp,factor=4)
    hp,Vhp=s[V].split(hp,factor=4)
    hpo,hpi=s[V].split(hp,factor=4)
    s[V].bind(wpo, block_x)
    s[V].bind(hpo, block_y)
    s[V].bind(cpo, block_z)
    s[V].bind(wpi, thread_x)
    s[V].bind(hpi, thread_y)
    s[V].bind(cpi, thread_z)
    s[V].reorder(cpo,cpi,hpo,hpi,wpi,wpo,Vp4,n)

    # Schedule BL local write
    s[VL].compute_at(s[V],n)
    _,_, hp, wp, p4  = s[VL].op.axis
    s[VL].reorder(hp,wp,p4)
    s[VL].vectorize(p4)
    k1,k2 = s[VL].op.reduce_axis
    #s[VL].unroll(wp)
    #s[VL].unroll(hp)
    #s[VL].unroll(k1)
    #s[VL].unroll(k2)

    s[WL].compute_at(s[VL], hp)
    _,_,hp, wp, p4 = s[WL].op.axis
    s[WL].vectorize(p4)  # vectorize memory load
    #s[WL].unroll(wp)
    #s[WL].unroll(hp)
    s[V].vectorize(Vp4)  # vectorize memory load
    #s[V].unroll(Vwp)  # vectorize memory load
    #s[V].unroll(Vhp)  # vectorize memory load
##========shedule end=================================
## shedule for M
##========shedule begin=================================
def s3(s, U, V, M):
    UL = s.cache_read(U, "local", [M])
    VL = s.cache_read(V, "local", [M])
    ML = s.cache_write(M, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis( "threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    n, kp, hp, wp, Mp4  = s[M].op.axis
    kpo,kpi=s[M].split(kp,factor=4)
    wp,Mwp=s[M].split(wp,factor=4)
    wpo,wpi=s[M].split(wp,factor=4)
    hpo,hpi=s[M].split(hp,factor=4)
    s[M].bind(wpo, block_x)
    s[M].bind(hpo, block_z)
    s[M].bind(kpo, block_y)
    s[M].bind(wpi, thread_x)
    s[M].bind(hpi, thread_z)
    s[M].bind(kpi, thread_y)
    s[M].reorder(kpo,hpo,hpi,wpi,wpo,Mwp,Mp4,kpi)

    # Schedule BL local write
    s[ML].compute_at(s[M],kpi)
    s[M].reorder(kpi, Mwp, Mp4)

    _,_, hp, wp, p4  = s[ML].op.axis
    rc, = s[ML].op.reduce_axis
    rco,rci = s[ML].split(rc,factor=4)
    s[ML].reorder(rco,wp,rci,p4)
    s[ML].vectorize(p4)
    #s[ML].unroll(wp)
    #s[ML].unroll(hp)
    #s[ML].unroll(k1)


    #schedule UL VL local read
    s[UL].compute_at(s[ML], rco)
    s[VL].compute_at(s[ML], rco)

    #split Ul VL workload
    a,b,hp, wp, _,p4 = s[UL].op.axis
    s[UL].vectorize(p4)  # vectorize memory load
    #s[UL].unroll(wp)
    #s[UL].unroll(hp)
    _,_,hp, wp, p4 = s[VL].op.axis
    s[VL].vectorize(p4)  # vectorize memory load
    #s[VL].unroll(wp)
    #s[VL].unroll(hp)

    s[M].vectorize(Mp4)  # vectorize memory load
    #s[M].unroll(Mwp)  # vectorize memory load
    #s[M].unroll(Mhp)  # vectorize memory load
##========shedule end=================================
#
##shedule for output
##========shedule begin=================================
def s4(s, A, M, output):
    O = output
    s[A].compute_inline()
    ML = s.cache_read(M, "local", [O])
    OL = s.cache_write(O, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    n, cp, hp, wp, Op4  = s[O].op.axis
    cpo,cpi=s[O].split(cp,factor=4)

    wp,Owp=s[O].split(wp,factor=4)
    wpo,wpi=s[O].split(wp,factor=4)

    hp,Ohp=s[O].split(hp,factor=4)
    hpo,hpi=s[O].split(hp,factor=4)

    s[O].bind(wpo, block_x)
    s[O].bind(hpo, block_y)
    s[O].bind(cpo, block_z)
    s[O].bind(wpi, thread_x)
    s[O].bind(hpi, thread_y)
    s[O].bind(cpi, thread_z)
    s[O].reorder(cpo,cpi,hpo,hpi,wpi,wpo,Owp,Ohp,Op4,n)

    # Schedule BL local write
    s[OL].compute_at(s[O],n)
    _,_, hp, wp, p4  = s[OL].op.axis
    s[OL].reorder(hp,wp,p4)
    s[OL].vectorize(p4)
    k1,k2 = s[OL].op.reduce_axis
    #s[OL].unroll(wp)
    #s[OL].unroll(hp)
    #s[OL].unroll(k1)
    #s[OL].unroll(k2)

    s[ML].compute_at(s[OL], hp)
    _,_,hp, wp, p4 = s[ML].op.axis
    s[ML].vectorize(p4)  # vectorize memory load
    #s[ML].unroll(wp)
    #s[ML].unroll(hp)
    s[O].vectorize(Op4)  # vectorize memory load

    #s[O].unroll(Owp)  # vectorize memory load
    #s[O].unroll(Ohp)  # vectorize memory load
##========shedule end=================================
O = output
A, M = s[O].op.input_tensors
U, V = s[M].op.input_tensors
G, filter_n = s[U].op.input_tensors
B, DataPad = s[V].op.input_tensors
Data, = s[DataPad].op.input_tensors
def shecule_default():
    s1(s, G, filter_n, U)
    s2(s, B, DataPad, V)
    s3(s, U, V, M)
    s4(s, A, M, output)

##========shedule begin=================================

##### space definition begin #####
cfg = tvm.autotvm.get_config()
cfg.define_split("inv_cp", oc_chunk, num_outputs=2, max_factor=8)
cfg.define_split("inv_wp", NTILE, num_outputs=3,
                    filter=lambda x: x.size[-1] == 4)
cfg.define_split("inv_hp", winop2_tile_size, num_outputs=3,
                    filter=lambda x: x.size[-1] == 4)

cfg.define_split("kernel_cp", CI, num_outputs=2, max_factor=32)
cfg.define_split("kernel_kp", oc_chunk, num_outputs=2, max_factor=32)

cfg.define_split("data_cp", oc_chunk, num_outputs=2, max_factor=32)
cfg.define_split("data_wp", oc_chunk, num_outputs=3,
                    filter=lambda x: x.size[-1] % 4 == 0)
cfg.define_split("data_hp", oc_chunk, num_outputs=3,
                    filter=lambda x: x.size[-1] % 4 == 0)

cfg.define_split("bgemm_kp", oc_chunk, num_outputs=2, max_factor=32)
cfg.define_split("bgemm_wp", oc_chunk, num_outputs=3,
                    filter=lambda x: x.size[-1] % 4 == 0)
cfg.define_split("bgemm_hp", oc_chunk, num_outputs=2)
##### space definition end #####

#tune_winograd._schedule_winograd_nchwc_io(cfg, s, output.op)
shecule_default()
##========shedule end=================================
#============for andriod=====================
arch = "arm64"
target_host = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)
target = tvm.target.Target("opencl -device=mali")

# Also replace this with the device key in your tracker
device_key = "MaliG72"
use_android=True

#===============anriod end==================
func = tvm.build(s, [Data, filter_n, output],
                 target=target,
                 target_host=target_host)
#assert func
#print(tvm.lower(s, [M,output], simple_mode=True))
#print(tvm.lower(s, [filter_n,Data,output], simple_mode=True))
with open('dd.cl','w') as fp:
    print(func.imported_modules[0].get_source(), file=fp) if len(
        func.imported_modules) > 0 else print("source not imported", file=fp)
dsp = tuple(int(i) for i in Data.shape)
fsp = tuple(int(i) for i in filter_n.shape)
osp = tuple(int(i) for i in output.shape)
###############test data=====================
#########andriod======================
tmp = tempdir()
if use_android:
    from tvm.contrib import ndk
    filename = "net.so"
    func.export_library(tmp.relpath(filename), ndk.create_shared)
else:
    filename = "net.tar"
    func.export_library(tmp.relpath(filename))

# upload module to device
print("Upload...")
remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9090,
                                        timeout=10000)
remote.upload(tmp.relpath(filename))
func = remote.load_module(filename)

# upload parameters to device
ctx = remote.context(str(target), 0)
#===============================================
a_np = np.arange(np.prod(dsp))
PACK4=4
H_P =img_H
W_P =img_W
C_P = CI//4
in_channel=CI
out_channel=CO
K_P=CO//4
a_np = a_np.reshape(1, C_P*PACK4, H_P, W_P)
a_np_t = a_np

w_np_t = np.arange(in_channel*out_channel*9).reshape(out_channel,in_channel,3,3)
w_np = w_np_t
#==A-tvm data prepare
a_np_tvm=np.ndarray(0)
for i in range(0,in_channel,4):
    b=a_np[:,i:i+4,:,:].transpose(0,2,3,1)
    ct=np.expand_dims(b,axis=1)
    if len(a_np_tvm.shape)!=5:
        a_np_tvm=ct
    else:
        a_np_tvm=np.concatenate((a_np_tvm,ct),axis=1)

a_np_tvm =  a_np_tvm*1.0
#==W-tvm data prepare
w_np_tvm=np.ndarray(0)
for i in range(0,out_channel,4):
    b=w_np[:,i:i+4,:,:].transpose(0,2,3,1)
    ct=np.expand_dims(b,axis=1)
    if len(w_np_tvm.shape)!=5:
        w_np_tvm=ct
    else:
        w_np_tvm=np.concatenate((w_np_tvm,ct),axis=1)
w_np_tvm = np.expand_dims(w_np_tvm, axis=-2)

w_np_tvm =  w_np_tvm* 1.0

##################=========end================
strides=1
padding=1

c_np = conv2d_nchw_python(a_np_t, w_np_t, strides, padding)
#=============##############
#data_tvm = tvm.nd.array(numpy.random.rand(*dsp).astype('float32'), ctx,dtype=ddtype)
#filter_tvm = tvm.nd.array(numpy.random.rand(
#    *fsp).astype('float32'), ctx, dtype=ddtype)
data_tvm=tvm.nd.array(a_np_tvm, ctx,dtype=ddtype)
filter_tvm = tvm.nd.array(w_np_tvm, ctx,dtype=ddtype)
output_tvm = tvm.nd.empty(osp, ctx=ctx, dtype=ddtype.replace('r','w'))
func(data_tvm, filter_tvm, output_tvm)
#================validate===================
#==C-tvm data prepare
c_tvm_o = output_tvm.asnumpy()

c_np_v = c_np.T.reshape(K_P*H_P*W_P, 4)
B1 = c_np_v[0::K_P, :]
for i in range(K_P-1):
    B1 = np.vstack((B1, c_np_v[i+1::K_P, :]))
c_np_v = B1.reshape(1, K_P, H_P, W_P, PACK4) * 1.0
#tvm.testing.assert_allclose(c_np_v, c_tvm_o, rtol=1e-2)
#exit(0)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
gflops = numpy.prod(np.array(osp[2:4] + fsp).astype('float')) / 1e9 * 2
t_cost = evaluator(data_tvm,filter_tvm,output_tvm).mean
print("Convolution: %f ms %f GFLOPS" % (t_cost*1e3,gflops/t_cost))
