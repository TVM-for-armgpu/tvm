import numpy as np
import tvm
import os
from tvm import te

print(os.getpid())
# The sizes of inputs and filters
batch = 1
in_channel = 1024
out_channel = 1024
in_size = 7
kernel = 1
pad = 0
stride = 1

open_image=1
ddtype='float32'
if open_image == 1:
    ddtype = "climgfloatr32"

# Algorithm
A = te.placeholder((batch, in_channel//4, in_size, in_size, 4), dtype=ddtype,name="A")
#W = te.placeholder((kernel, kernel, in_channel, out_channel), dtype=ddtype, name="W")
ry = te.reduce_axis((0, in_size), name="ry")
rx = te.reduce_axis((0, in_size), name="rx")
# Pad input
Pool = te.compute(
    (batch, in_channel // 4, 1, 1, 4),
    lambda nn, cc, yy, xx, bnn: te.sum(A[nn, cc, yy, xx, bnn], axis=[ry, rx]),
    name="B",
)
s = te.create_schedule(Pool.op)

num_thread = 8
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
Out=Pool
if len(s[Out].op.axis) == 5:
    Pool_in = Pool.op.input_tensors[0]
    #Pool_inL = s.cache_read(Pool_in, "local", [Pool])
    n, cp, hp, wp, bcn = s[Out].op.axis
else:
    n, cp, hp, wp = s[Out].op.axis

if Pool.op in s.outputs:
    Out = Pool
    OL = s.cache_write(Pool, "local")

by, ty = s[Out].split(n, factor=num_thread)
bx, tx = s[Out].split(cp, factor=num_thread)

s[Out].reorder(by, bx, ty, tx, hp, wp, bcn)

s[Out].bind(ty, thread_y)
s[Out].bind(tx, thread_x)
s[Out].bind(by, block_y)
s[Out].bind(bx, block_x)
if Pool.op in s.outputs:
    s[OL].compute_at(s[Out], tx)
    _,cp,h,w,bn4 = s[OL].op.axis
    s[OL].reorder(cp, h, w, bn4)
    s[OL].vectorize(bn4)
else:
    s[Pool].compute_at(s[Out], tx)
#s[Pool_inL].compute_at(s[OL], cp)
#_, _, _, _, inp_bcn = s[Pool_inL].op.axis
#s[Pool_inL].vectorize(inp_bcn)

s[Out].vectorize(bcn)
print(tvm.lower(s, [A,Out], simple_mode=True))