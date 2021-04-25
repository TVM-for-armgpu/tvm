from os import environ

import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils



HOST = environ["DEVICE_THETHERING_IP"]
PORT = 9090
REMOTE = rpc.connect(HOST, PORT)
CTXT = REMOTE.cl()

class DeviceFunction:
    def __init__(self, host_fn):
        self.srcs = [x.get_source() for x in host_fn.imported_modules]
        self.host_fn = host_fn

        temp = utils.tempdir()
        path = temp.relpath("lib_cl.stackvm")
        host_fn.export_library(path)
        REMOTE.upload(path)
        self.dev_fn = REMOTE.load_module("lib_cl.stackvm")
    def __call__(self, *args, **kwargs):
        return self.dev_fn(*args, **kwargs)

def host2dev(host_fn):
    """
    Upload schedule function to device and bind the generated sources to the
    remote function object.
    """
    def inner(*args, **kwargs):
        return DeviceFunction(host_fn(*args, **kwargs))
    return inner





access_reg = []
cur_access_reg = []

last_for_op = [None]



def is_nested_for(outer, inner):
    def visitor(op):
        if isinstance(op, inner):
            visitor.found = True
    v = visitor
    v.found = False

    tvm.tir.stmt_functor.post_order_visit(outer.body, visitor)

def find_allocation(op):
    global cur_access_reg, last_for_op
    print(f"found op {type(op)}")

    if isinstance(op, tvm.tir.For):
        access_reg.append(cur_access_reg)
        cur_access_reg = []
        if last_for_op[0] is not None:
            last_for_op
        last_for_op = [op]

    elif isinstance(op, tvm.tir.Load):
        print("load var name is:", op.buffer_var.name)
        cur_access_reg.append(op.buffer_var.name)

    elif isinstance(op, tvm.tir.Store):
        print("store var name is:", op.buffer_var.name)
        cur_access_reg.append(op.buffer_var.name)

@tvm.tir.transform.prim_func_pass(opt_level=0)
def analyze(f, mod, ctx):
    tvm.tir.stmt_functor.post_order_visit(f.body, find_allocation)
    print(access_reg)
    return f





@host2dev
def matmul(M, K, N, A, B):
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(A[m, k] * B[k, n], axis=k),
        name="C",
    )
    s = te.create_schedule(C.op)

    # Lower to TIR and optimize.
    with tvm.transform.PassContext():
        print(tvm.lower(s, [A, B, C]))

    # Define intrinsic binding.
    m_group_id, m_local_id = s[C].split(C.op.axis[0], factor=32)
    n_group_id, n_local_id = s[C].split(C.op.axis[1], factor=32)

    s[C].bind(m_group_id, te.thread_axis("blockIdx.x"))
    s[C].bind(m_local_id, te.thread_axis("threadIdx.x"))

    s[C].bind(n_group_id, te.thread_axis("blockIdx.y"))
    s[C].bind(n_local_id, te.thread_axis("threadIdx.y"))

    func = tvm.build(s, [A, B, C], "opencl")

    return func





def main():
    # Define schedule.
    M = tvm.runtime.convert(32)
    K = tvm.runtime.convert(64)
    N = tvm.runtime.convert(128)
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")

    dev_func = matmul(M, K, N, A, B)

    # Preview generated CL source.
    for src in dev_func.srcs:
        print(src)

    # Run & test.
    A_data = tvm.nd.array(np.random.uniform(size=(M.value * K.value)).astype(A.dtype).reshape(M.value, K.value), CTXT)
    B_data = tvm.nd.array(np.random.uniform(size=(K.value * N.value)).astype(B.dtype).reshape(K.value, N.value), CTXT)
    C_data = tvm.nd.array(np.zeros((M.value * N.value), dtype="float32").reshape(M.value, N.value), CTXT)

    dev_func(A_data, B_data, C_data)

    np.testing.assert_almost_equal(C_data.asnumpy(), np.dot(A_data.asnumpy(), B_data.asnumpy()), decimal=5)
    print("OpenCL test passed!")



if __name__ == "__main__":
    main()
