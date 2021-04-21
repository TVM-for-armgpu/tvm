from os import environ

import numpy as np

import tvm
from tvm import te
from tvm import rpc
from tvm.contrib import utils


ARCH_DETAIL = {
    "max_local_memory_per_block": 181 * 384,
    "max_shared_memory_per_block": 32768,
    "max_threads_per_block": 384,
    "max_thread_x": 384,
    "max_thread_y": 384,
    "max_thread_z": 384,
    "max_vthread": 1,
    "max_vector_bytes": 16,
    "buf_top_cache_bytes": 65536,
    "img_top_cache_bytes": 1024,
}


HOST = environ["DEVICE_THETHERING_IP"]
PORT = 9090
REMOTE = rpc.connect(HOST, PORT)
CTXT = REMOTE.cl()

CTXT.set_arch_detail(ARCH_DETAIL)

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

    # Define intrinsic binding.

    m_item_id = C.op.axis[0]
    k_item_id = s[C].op.reduce_axis[0]
    n_item_id = C.op.axis[1]

    m_vec_id, m_comp_id = s[C].split(m_item_id, factor=4)
    m_group_id, m_local_id = s[C].split(m_vec_id, factor=8)

    k_vec_id, k_comp_id = s[C].split(k_item_id, factor=4)

    n_global_id, n_unroll_id = s[C].split(n_item_id, factor=4)
    n_group_id, n_local_id = s[C].split(n_global_id, factor=8)

    s[C].reorder(
        n_group_id,
        n_local_id,
        m_group_id,
        m_local_id,
        k_vec_id,
        k_comp_id,
        n_unroll_id,
        m_comp_id,
    )

    s[C].vectorize(m_comp_id)
    s[C].unroll(k_comp_id)
    s[C].unroll(n_unroll_id)
    s[C].bind(m_group_id, te.thread_axis("blockIdx.x"))
    s[C].bind(n_group_id, te.thread_axis("blockIdx.y"))
    s[C].bind(m_local_id, te.thread_axis("threadIdx.x"))
    s[C].bind(n_local_id, te.thread_axis("threadIdx.y"))

    func = tvm.build(s, [A, B, C], "opencl", name="matmul")

    return func





def main():
    print(111)
    # Define schedule.
    M = tvm.runtime.convert(1024)
    K = tvm.runtime.convert(1024)
    N = tvm.runtime.convert(1024)

    dev_func = matmul(M, K, N)

    # Preview generated CL source.
    #for src in dev_func.srcs:
    #    print(src)

    # Run & test.
    A_data = tvm.nd.array(np.random.uniform(size=(K.value * M.value)).astype("float32").reshape(K.value, M.value), CTXT)
    B_data = tvm.nd.array(np.random.uniform(size=(K.value * N.value)).astype("float32").reshape(K.value, N.value), CTXT)
    C_data = tvm.nd.array(np.zeros((M.value * N.value), dtype="float32").reshape(M.value, N.value), CTXT)
    print(222)

    dev_func(A_data, B_data, C_data)

    np.testing.assert_almost_equal(C_data.asnumpy(), np.dot(A_data.asnumpy().T, B_data.asnumpy()), decimal=3)
    print("OpenCL test passed!")

    print(f"Kernel latency: {dev_func.time(A_data, B_data, C_data)}")



if __name__ == "__main__":
    main()
