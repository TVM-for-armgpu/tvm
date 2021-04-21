from os import environ

import numpy as np

import tvm
from tvm import autotvm
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
        self.timer = self.dev_fn.time_evaluator(self.dev_fn.entry_name, CTXT, number=10)
    def __call__(self, *args, **kwargs):
        return self.dev_fn(*args, **kwargs)
    def time(self, *args):
        return self.timer(*args).mean

def host2dev(host_fn):
    """
    Upload schedule function to device and bind the generated sources to the
    remote function object.
    """
    def inner(*args, **kwargs):
        return DeviceFunction(host_fn(*args, **kwargs))
    return inner





@tvm.tir.transform.prim_func_pass(opt_level=0)
def analyze(f, mod, ctx):
    class AccessFootPrint:
        def __init__(self, bufload_op):
            self.name = bufload_op.buffer.name
            self.idxs = bufload_op.indices
            self.dtype = bufload_op.buffer.dtype

        def __repr__(self):
            return f"{self.name}{self.idxs}: {self.dtype}"

    class Visitor:
        def __init__(self):
            self.access_reg = []
            self.cur_access_reg = []

        def __call__(self, op):
            print(f"found op {type(op)}")

            if isinstance(op, tvm.tir.BufferLoad):
                self.cur_access_reg += [AccessFootPrint(op)]

            elif isinstance(op, tvm.tir.BufferStore):
                var_name = op.buffer.name
                self.access_reg += [(var_name, self.cur_access_reg)]
                self.cur_access_reg = []

    v = Visitor()
    tvm.tir.stmt_functor.post_order_visit(f.body, v)
    print(v.access_reg)

    return f





@autotvm.template("matmul")
def matmul(M, K, N, dtype="float32"):
    M = tvm.runtime.convert(M)
    K = tvm.runtime.convert(K)
    N = tvm.runtime.convert(N)

    # Input tensors.
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)

    # Computation and output tensor.
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(A[m, k] * B[k, n], axis=k),
        name="C",
    )
    s = te.create_schedule(C.op)

    # Extract schedule axes.
    M = C.op.axis[0]
    K = k
    N = C.op.axis[1]

    # Define axis-split.
    cfg = autotvm.get_config()
    cfg.define_split("M", M, num_outputs=5)
    cfg.define_split("K", K, num_outputs=2)
    cfg.define_split("N", N, num_outputs=4)

    m_group_id, m_local_id, m_unroll_id, m_vec_id, m_comp_id = cfg["M"].apply(s, C, M)
    k_vec_id, k_comp_id = cfg["K"].apply(s, C, K)
    n_group_id, n_local_id, n_unroll_id, n_item_id = cfg["N"].apply(s, C, N)

    s[C].reorder(
        n_group_id,
        n_local_id,
        n_unroll_id,
        n_item_id,
        m_group_id,
        m_local_id,
        m_unroll_id,
        m_vec_id,
        m_comp_id,
    )

    s[C].storage_align(M, 4, 0)
    s[C].vectorize(m_comp_id)
    s[C].unroll(k_comp_id)
    s[C].bind(m_group_id, te.thread_axis("blockIdx.x"))
    s[C].bind(n_group_id, te.thread_axis("blockIdx.y"))
    s[C].bind(m_local_id, te.thread_axis("threadIdx.x"))
    s[C].bind(n_local_id, te.thread_axis("threadIdx.y"))

    # Lower to TIR and optimize.
    with tvm.transform.PassContext(config={"tir.add_lower_pass": [(0, analyze)]}):
        print(tvm.lower(s, [A, B, C]))

    return s, [A, B, C]





def main():
    M = 32
    K = 64
    N = 128

    task = autotvm.task.create("matmul", args=(N, K, M, "float32"), target="opencl")
    print(task.config_space)

if __name__ == "__main__":
    main()
