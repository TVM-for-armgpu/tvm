from os import environ
from sys import argv

import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils, stackvm

HOST = environ["DEVICE_THETHERING_IP"]
PORT = 9090
print(0)
TRACKER = rpc.connect_tracker("127.0.0.1", PORT)
print(argv[1])
REMOTE = TRACKER.request(argv[1] or "", priority=1, session_timeout=5)
print(2)
CTXT = REMOTE.cl()

class DeviceFunction:
    def __init__(self, host_fn):
        self.srcs = [x.get_source() for x in host_fn.imported_modules]
        self.host_fn = host_fn

        temp = utils.tempdir()
        path = temp.relpath("lib_cl.stackvm")
        host_fn.export_library(path, stackvm.stackvm)
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






@host2dev
def matmul(M, K, N):
    A = te.placeholder((K, M), name="A")
    B = te.placeholder((K, N), name="B")

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda m, n: te.sum(A[k, m] * B[k, n], axis=k),
        name="C",
    )
    s = te.create_schedule(C.op)
    AA = s.cache_read(A, "local", [C])
    BB = s.cache_read(B, "local", [C])
    CC = s.cache_write(C, "local")

    m_item_id, n_item_id = C.op.axis
    m_global_id, m_unroll_id = s[C].split(m_item_id, factor=2)
    n_global_id, n_unroll_id = s[C].split(n_item_id, factor=2)

    m_group_id, m_local_id = s[C].split(m_global_id, factor=2)
    n_group_id, n_local_id = s[C].split(n_global_id, factor=2)

    s[C].reorder(
        n_group_id,
        m_group_id,
        n_local_id,
        m_local_id,
        n_unroll_id,
        m_unroll_id,
    )

    s[CC].compute_at(s[C], m_local_id)
    s[C].bind(m_group_id, te.thread_axis("blockIdx.x"))
    s[C].bind(n_group_id, te.thread_axis("blockIdx.y"))
    s[C].bind(m_local_id, te.thread_axis("threadIdx.x"))
    s[C].bind(n_local_id, te.thread_axis("threadIdx.y"))

    s[C].unroll(m_unroll_id)
    s[C].vectorize(n_unroll_id)

    k_item_id = CC.op.reduce_axis[0]
    s[AA].compute_at(s[CC], k_item_id)
    s[BB].compute_at(s[CC], k_item_id)

    m_unroll_id, n_unroll_id = CC.op.axis
    s[AA].vectorize(s[AA].op.axis[-1])
    s[BB].vectorize(s[BB].op.axis[-1])

    s[CC].reorder(
        k_item_id,
        n_unroll_id,
        m_unroll_id,
    )
    s[CC].unroll(m_unroll_id)
    s[CC].vectorize(n_unroll_id)

    #print(tvm.lower(s, [A, B, C])); assert False

    func = tvm.build(s, [A, B, C], "opencl", name="matmul")

    return func





def main():
    # Define schedule.
    M = tvm.runtime.convert(8)
    K = tvm.runtime.convert(8)
    N = tvm.runtime.convert(8)

    dev_func = matmul(M, K, N)

    # Preview generated CL source.
    for src in dev_func.srcs:
        print(src)

    # Run & test.
    A_data = tvm.nd.array(np.random.uniform(size=(K.value * M.value)).astype("float32").reshape(K.value, M.value), CTXT)
    B_data = tvm.nd.array(np.random.uniform(size=(K.value * N.value)).astype("float32").reshape(K.value, N.value), CTXT)
    C_data = tvm.nd.array(np.zeros((M.value * N.value), dtype="float32").reshape(M.value, N.value), CTXT)

    dev_func(A_data, B_data, C_data)

    np.testing.assert_almost_equal(C_data.asnumpy(), np.dot(A_data.asnumpy().T, B_data.asnumpy()), decimal=3)
    print("OpenCL test passed!")

    print(f"Kernel latency: {dev_func.time(A_data, B_data, C_data)}")



if __name__ == "__main__":
    main()
