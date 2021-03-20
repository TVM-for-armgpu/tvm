import numpy as np
import tvm
import os
from tvm import te
import sys
sys.path.append('..')

print(os.getpid())
# The sizes of inputs and filters
batch = 1
in_channel = 36
out_channel =32
in_size =101
kernel = 1
pad = 0
stride = 1

open_image=0
ddtype='float32'
if open_image == 1:
    ddtype = "climgfloatr32"

# Algorithm
PACK4 = 4
#PACK2=2
PACK2=1
#W_P = (in_size+1)//2*2//PACK2
#H_P = (in_size+1)//2*2//PACK2
W_P=H_P=in_size
C_P = (in_channel+3)//PACK4
K_P=out_channel//PACK4

#A = te.placeholder((C_P,H_P,PACK2,W_P,PACK2,PACK4), dtype=ddtype,name="A")
A = te.placeholder((C_P,H_P,W_P,PACK4), dtype=ddtype,name="A")
W = te.placeholder((C_P*PACK4,out_channel), dtype=ddtype, name="W")
out_size = in_size
# Pad input
Apad = A
# Create reduction variables
rc = te.reduce_axis((0, C_P*PACK4), name="rc")
# Compute the convolution
idxdiv = tvm.tir.indexdiv
idxmod = tvm.tir.indexmod
B = te.compute(
    (K_P, H_P,W_P,PACK4),
    #lambda ff,yy,xx_p4: extern_op.mysum(
    lambda ff,yy,xx,p4: te.sum(
        #extern_op.mymul(Apad[rc//4,yy,rc%4+xx_p4//4*4] , W[rc,ff*PACK4+idxmod(xx_p4,4)]), axis=[rc]
        Apad[rc//4,yy,xx,rc%4] * W[rc,ff*PACK4+p4], axis=[rc]
    ),
    name="B",
)
if open_image ==1:
    B.dtype="climgfloatw32"

# Designate the memory hierarchy
s = te.create_schedule(B.op)
#s[Apad].compute_inline()  # compute Apad inline
#AA = s.cache_read(Apad, "shared", [B])
#WW = s.cache_read(W, "shared", [B])
AL = s.cache_read(Apad, "local", [B])
WL = s.cache_read(W, "local", [B])
BL = s.cache_write(B, "local")

# tile consts
tile = 1
num_thread = 4
block_factor = tile * num_thread
step = 4
vthread = 2

# Get the GPU thread indices
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis( "threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")
thread_z = te.thread_axis("threadIdx.z")
thread_xz = te.thread_axis("vthread", name="vx")
thread_yz = te.thread_axis("vthread", name="vy")

# Split the workloads
kp, hp, wp,p4  = s[B].op.axis
wp,wp4 = wp,p4
wp,wpl=s[B].split(wp,factor=2)
wp4=s[B].fuse(wp4,wpl)
hpo,hpii=s[B].split(hp,factor=2)


#s[B].reorder(wpo,hpo,kp,wpi,hpi,p4)
#whp = s[B].fuse(wpi,hpi)
#s[B].reorder(wpo,hpo,kp,whp,p4)
#s[B].vectorize(p4)  # vectorize memory load
#s[B].unroll(whp)
#rc, = s[B].op.reduce_axis
#rco,rci=s[B].split(rc,factor=4)
#s[B].reorder(wpo,hpo,kp,rco,whp,rci,p4)
#s[B].vectorize(p4)  # vectorize memory load
#s[B].unroll(whp)
#
#print(tvm.lower(s, [A, W, B], simple_mode=True))
#exit(0)
########
hpo,hpi=s[B].split(hpo,factor=block_factor)
wpo,wpi=s[B].split(wp,factor=block_factor//2)
kpo,kpi=s[B].split(kp,factor=block_factor)

#target = 'llvm -mcpu=core-avx2'
#func = tvm.build(s, [A, W, B], target=target, name='mmult')
#np.random.seed(5)
#a_np = np.random.uniform(size=(C_P,H_P,W_P,PACK4)).astype("float32")
#w_np = np.random.uniform(size=(in_channel,out_channel)).astype("float32")
#ctx = tvm.context(target, 0)
#a = tvm.nd.array(a_np, ctx,dtype=A.dtype)
#w = tvm.nd.array(w_np, ctx,dtype=W.dtype)
#b = tvm.nd.array(np.zeros((K_P, H_P,W_P,PACK4), dtype="float32"), ctx, dtype=B.dtype)
#func(a, w, b)
#exit(0)
#
# Bind the iteration variables to GPU thread indices

s[B].bind(hpo, block_y)
s[B].bind(wpo, block_x)
s[B].bind(kpo, block_z)
s[B].bind(hpi, thread_y)
s[B].bind(wpi, thread_x)
s[B].bind(kpi, thread_z)
s[B].reorder(wpi,hpo,kpo,hpi,wpo,hpii,kpi)


# Schedule BL local write
s[BL].compute_at(s[B],kpi)
s[B].reorder(kpi,hpii)
kp, hp, wp, p4  = s[BL].op.axis
#wpo,wpi=s[BL].split(wp,factor=2)
#hpo,hpi=s[BL].split(hp,factor=2)

#s[B].reorder(wpo,hpo,kp,wpi,hpi,p4)

s[BL].reorder(hp,wp,p4)
whp = s[BL].fuse(wp,hp)
rc, = s[BL].op.reduce_axis
rco,rci=s[BL].split(rc,factor=4)
s[BL].reorder(rco,rci,whp,p4)
s[BL].vectorize(p4)  # vectorize memory load
#s[BL].unroll(whp)
#s[BL].unroll(p4)
#s[BL].unroll(rci)

# Attach computation to iteration variables
#s[AA].compute_at(s[BL], rco)
#s[WW].compute_at(s[BL], rco)
s[AL].compute_at(s[BL], rco)
s[WL].compute_at(s[BL], rco)

kp, hp, wp,p4 = s[AL].op.axis
#ty, ci = s[AA].split(ci, nparts=num_thread)
#tx, ni = s[AA].split(ni, nparts=num_thread)
wpo, wpi = wp,p4
#s[AA].reorder(ty, tx, yi, xi, ci, ni)
#s[AA].bind(ty, thread_y)
#s[AA].bind(tx, thread_x)
s[AL].vectorize(wpi)  # vectorize memory load
s[AL].unroll(wpo)
s[AL].unroll(hp)
#s[AL].reorder(wpo,hp)

# Schedule for W's shared memory load
kp, cp = s[WL].op.axis
print(kp,'==========')
#ty, ci = s[WL].split(ci, nparts=num_thread)
_, cpi = s[WL].split(cp, factor=4)
#tx, fi = s[WW].split(fi, nparts=num_thread)
#s[WW].reorder(ty, tx, yi, xi, ci, fi)
#s[WW].bind(ty, thread_y)
#s[WW].bind(tx, thread_x)
s[WL].vectorize(cpi)  # vectorize memory load
s[WL].unroll(kp)

wpio,wpii = s[B].split(wp4, factor=4)
s[B].vectorize(wpii)  # vectorize memory load
#s[B].unroll(wpio)
#s[B].unroll(hpii)


#print(tvm.lower(s, [A, W, B], simple_mode=True))
#exit(0)


###############################################################################
# Generate CUDA Kernel
# --------------------
#
# Finally we use TVM to generate and compile the CUDA kernel, and evaluate the
# latency of convolution.
#
#print(tvm.lower(s, [A, W, B], simple_mode=True))

target="opencl"
#target="llvm"
#target="cuda"
func = tvm.build(s, [A, W, B], target)
print("------opencl code------")
print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
ctx = tvm.context(target, 0)
#ctx = tvm.gpu(0)

H_P=W_P=in_size
ao_np = np.arange(in_channel*in_size*in_size)
#a_np=np.zeros((C_P*PACK4,H_P,W_P))
#a_np[:in_channel,:in_size,:in_size]=ao_np.reshape(in_channel,in_size,in_size)
a_np=ao_np.reshape(in_channel,in_size,in_size)
#a_np=a_np.reshape(C_P*PACK4,H_P*W_P)
a_np=a_np.reshape(C_P*PACK4,H_P*W_P)
wo_np = np.arange(in_channel*out_channel).reshape(out_channel,in_channel)
w_np=np.zeros((out_channel,C_P*PACK4))
w_np[:out_channel,:in_channel]=wo_np
w_np=wo_np

a_tvm=a_np.T.reshape(C_P*H_P*W_P,4)
B1=a_tvm[0::C_P,:]
for i in range(C_P-1):
    B1=np.vstack((B1,a_tvm[i+1::C_P,:]))
a_tvm=B1.reshape(C_P,H_P,W_P,PACK4)*1.0
w_tvm=w_np.T*1.0

a = tvm.nd.array(a_tvm ,ctx,dtype=A.dtype)
w = tvm.nd.array(w_tvm ,ctx,dtype=W.dtype)
b = tvm.nd.array(np.zeros((K_P, H_P,W_P,PACK4), dtype="float32"), ctx, dtype=B.dtype)
func(a, w, b)
A=a_np.astype("float32")
W=w_np.astype("float32")
C=W.dot(A)
C=C.reshape(K_P*PACK4,H_P*W_P).T.reshape(K_P*H_P*W_P,4)
B1=C[0::K_P,:]
for i in range(K_P-1):
    B1=np.vstack((B1,C[i+1::K_P,:]))
C=B1.reshape(K_P,H_P*W_P*PACK4)
Ctvm = b.asnumpy().reshape(K_P,H_P*W_P*PACK4)
print(C.shape)
print(Ctvm.shape)
np.testing.assert_allclose(C, Ctvm, rtol=1e-2)
np.savetxt("filename.txt",C)
evaluator = func.time_evaluator(func.entry_name, ctx, number=15)
cost = evaluator(a, w, b).mean
print("\nanswer::correct!,Convolution: %fms, about %f GFLOPS" % (cost*1e3,out_channel*in_channel*in_size*in_size*2/cost/1e9))
