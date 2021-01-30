from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np
n = te.var("n")
m = te.var("m")
A = te.placeholder((n, m), name="A")
k = te.reduce_axis((0, m), "k")
B = te.compute((n,), lambda i: te.sum(A[i, k], axis=k), name="B")

s = te.create_schedule(B.op)
ko, ki = s[B].split(B.op.reduce_axis[0], factor=16)
BF = s.rfactor(B, ki)

######################################################################
# The scheduled operator of B also get rewritten to be sum over
# the first axis of reduced result of B.f
#
print(s[B].op.body)

######################################################################
# Cross Thread Reduction
# ----------------------
# We can now parallelize over the factored axis.
# Here the reduction axis of B is marked to be a thread.
# TVM allows reduction axis to be marked as thread if it is the only
# axis in reduction and cross thread reduction is possible in the device.
#
# This is indeed the case after the factoring.
# We can directly compute BF at the reduction axis as well.
# The final generated kernel will divide the rows by blockIdx.x and threadIdx.y
# columns by threadIdx.x and finally do a cross thread reduction over threadIdx.x
#
xo, xi = s[B].split(s[B].op.axis[0], factor=32)
xo,xoo=s[B].split(xo, factor=4)
s[B].bind(xo, te.thread_axis("blockIdx.x"))
s[B].bind(xi, te.thread_axis("threadIdx.y"))
tx = te.thread_axis("threadIdx.x")
s[B].bind(s[B].op.reduce_axis[0], tx)
#s[B].reorder(B.op.reduce_axis[0],xoo)
s[BF].compute_at(s[B], s[B].op.reduce_axis[0])
s[B].set_store_predicate(tx.var.equal(0))
target='cuda'
fcuda = tvm.build(s, [A, B], target)
print(fcuda.imported_modules[0].get_source())

######################################################################
# Verify the correctness of result kernel by comparing it to numpy.
#
nn = 128
ctx = tvm.context(target, 0)
ctx = tvm.gpu(0)
a = tvm.nd.array(np.random.uniform(size=(nn, nn)).astype(A.dtype), ctx)
b = tvm.nd.array(np.zeros(nn, dtype=B.dtype), ctx)
fcuda(a, b)
tvm.testing.assert_allclose(b.asnumpy(), np.sum(a.asnumpy(), axis=1), rtol=1e-4)

