
import numpy as np
import tvm
import os
from tvm import te

print(os.getpid())
# The sizes of inputs and filters
batch = 1
in_channel = 256
out_channel = 512
in_size = 32
kernel = 1
pad = 0
stride = 1

open_image=1
ddtype='float32'
if open_image == 1:
    ddtype = "climgfloatr32"

# Algorithm
A = te.placeholder((in_size, in_size, in_channel, batch), dtype=ddtype,name="A")
W = te.placeholder((kernel, kernel, in_channel, out_channel), dtype=ddtype, name="W")
out_size = (in_size - kernel + 2 * pad) // stride + 1
# Pad input
Apad = te.compute(
    (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
    lambda yy, xx, cc, nn: tvm.tir.if_then_else(
        tvm.tir.all(yy >= pad, yy - pad < in_size, xx >= pad, xx - pad < in_size),
        A[yy - pad, xx - pad, cc, nn],
        tvm.tir.const(0.0, "float32"),
    ),
    name="Apad",
)
# Create reduction variables
rc = te.reduce_axis((0, in_channel), name="rc")
ry = te.reduce_axis((0, kernel), name="ry")
rx = te.reduce_axis((0, kernel), name="rx")
# Compute the convolution
B = te.compute(
    (out_size, out_size, out_channel, batch),
    lambda yy, xx, ff, nn: te.sum(
        Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff], axis=[ry, rx, rc]
    ),
    name="B",
)
#if open_image ==1:
#    B.dtype="climgfloatw32"


###############################################################################
# Memory Hierarchy
# ----------------
#
# We first specify the memory hierarchy for buffers. The figure below shows the
# GPU memory hierarchy. One important difference from CPU memory hierarchy is
# that GPU provides a cache buffer called shared memory, which is managed by
# programmers. Thus how to maximize the data reuse in the shared memory is
# critical to achieve high performance in GPU kernels.
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/gpu_memory_hierarchy.png
#      :align: center
#      :height: 319px
#      :width: 271px
#
# In this example, we load both Apad and W into buffer AA and WW, which are
# stored in the shared memory. These bufferes will be later shared by all
# threads within the same thread block to compute the convolution. Each thread
# then loads its own part from shared buffer into their local registers, AL and
# WL. BL is a local cache of output B, which is also stored in the thread local
# registers.
#

# Designate the memory hierarchy
s = te.create_schedule(B.op)
s[Apad].compute_inline()  # compute Apad inline
AA = s.cache_read(Apad, "shared", [B])
WW = s.cache_read(W, "shared", [B])
AL = s.cache_read(AA, "local", [B])
WL = s.cache_read(WW, "local", [B])
BL = s.cache_write(B, "local")

###############################################################################
# Blocking
# --------
#
# The following code splits the workload into thread blocks and individual
# threads. We follow the blocking scheme in the matrix multiply. As shown in the
# figure below, given a pixel coordinate (y, x), a thread block is responsible
# for computing a region of block_factor x block_factor (64 x 64) for output
# channels and batch. Due to the limit of shared memory space, we only load step
# x block_factor (8 x 64) data from Apad and B each time to buffers in the
# shared memory.
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/conv_gpu_blocking.png
#      :align: center
#      :height: 308px
#      :width: 317px
#

# tile consts
tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2

# Get the GPU thread indices
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

# Split the workloads
hi, wi, fi, ni = s[B].op.axis
bz = s[B].fuse(hi, wi)
#s[B].vectorize(ni)  # vectorize memory load
by, fi = s[B].split(fi, factor=block_factor)
bx, ni = s[B].split(ni, factor=block_factor)
# Bind the iteration variables to GPU thread indices
s[B].bind(bz, block_z)
s[B].bind(by, block_y)
s[B].bind(bx, block_x)

###############################################################################
# Virtual Thread Split
# --------------------
#
# We further split the workload from a thread block to individual threads. To
# avoid *memory bank conflict*, we use virtual thread to split the area into 4
# parts, and then tile into 8x8 grids. Therefore, shown in the figure below,
# each thread computes 4 strided grids, where size of each grid is 4 x 4.
#
# .. image:: https://github.com/dmlc/web-data/raw/main/tvm/tutorial/conv_gpu_vthread.png
#      :align: center
#      :height: 188px
#      :width: 268px
#

#tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
#txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
ty, fi = s[B].split(fi, nparts=num_thread)
tx, ni = s[B].split(ni, nparts=num_thread)
#s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
s[B].reorder(bz, by, bx, ty, tx, fi, ni)

#s[B].bind(tyz, thread_yz)
#s[B].bind(txz, thread_xz)
s[B].bind(ty, thread_y)
s[B].bind(tx, thread_x)

###############################################################################
# Cooperative Fetching
# --------------------
#
# As mentioned before, each time step we need to transfer step x block_factor
# data from GPU global memory to shared memory. In order to reduce the memory
# transfer per thread, the following code lets threads in the same thread block
# coopertively fetch dependent data from global memory.
#


# Schedule BL local write
s[BL].compute_at(s[B], tx)
ni, noo = s[B].split(ni, factor=4)
s[B].vectorize(noo)  # vectorize memory load
yi, xi, fi, ni = s[BL].op.axis
ry, rx, rc = s[BL].op.reduce_axis
rco, rci = s[BL].split(rc, factor=step)
s[BL].reorder(rco, ry, rx, rci, fi, ni)

# Attach computation to iteration variables
s[AA].compute_at(s[BL], rx)
s[WW].compute_at(s[BL], rx)
s[AL].compute_at(s[BL], rci)
s[WL].compute_at(s[BL], rci)

# Schedule for A's shared memory load
yi, xi, ci, ni = s[AA].op.axis
#ty, ci = s[AA].split(ci, nparts=num_thread)
#tx, ni = s[AA].split(ni, nparts=num_thread)
#_, ni = s[AA].split(ni, factor=4)
_, ci = s[AA].split(ci, factor=4)
#s[AA].reorder(ty, tx, yi, xi, ci, ni)
#s[AA].bind(ty, thread_y)
#s[AA].bind(tx, thread_x)
#s[AA].vectorize(ni)  # vectorize memory load
s[AA].vectorize(ci)  # vectorize memory load

# Schedule for W's shared memory load
yi, xi, ci, fi = s[WW].op.axis
ty, ci = s[WW].split(ci, nparts=num_thread)
tx, fi = s[WW].split(fi, nparts=num_thread)
_, fi = s[WW].split(fi, factor=4)
s[WW].reorder(ty, tx, yi, xi, ci, fi)
s[WW].bind(ty, thread_y)
s[WW].bind(tx, thread_x)
s[WW].vectorize(fi)  # vectorize memory load


###############################################################################
# Generate CUDA Kernel
# --------------------
#
# Finally we use TVM to generate and compile the CUDA kernel, and evaluate the
# latency of convolution.
#
#print(tvm.lower(s, [A, W, B], simple_mode=True))

target="opencl"
#target="cuda"
func = tvm.build(s, [A, W, B], target)
print("------opencl code------")
print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
ctx = tvm.context(target, 0)
#ctx = tvm.gpu(0)
np.random.seed(5)
a_np = np.random.uniform(size=(in_size, in_size, in_channel, batch)).astype("float32")
w_np = np.random.uniform(size=(kernel, kernel, in_channel, out_channel)).astype("float32")
print(w_np.shape)
#for i in range(8):
#    for j in range(8):
#        a_np[i,j,0,0]=i
#
#for i in range(3):
#    for j in range(3):
#        w_np[i,j,0,0]=j
a = tvm.nd.array(a_np, ctx,dtype=A.dtype)
w = tvm.nd.array(w_np, ctx,dtype=W.dtype)
b = tvm.nd.array(np.zeros((out_size, out_size, out_channel, batch), dtype="float32"), ctx, dtype=B.dtype)
func(a, w, b)
np.savetxt("filename.txt",b.asnumpy()[:,:,0,0])
evaluator = func.time_evaluator(func.entry_name, ctx, number=5)
print("Convolution: %f ms" % (evaluator(a, w, b).mean * 1e3))
