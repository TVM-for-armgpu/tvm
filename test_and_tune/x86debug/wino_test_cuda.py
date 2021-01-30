import os
import numpy as np
import tvm
from tvm import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi import util
from topi.nn import pad

def reference_direct(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')

    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype
    dilation = 1

    @memoize("topi.tests.test_topi_conv2d_nchw.reference_direct")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    ctx = tvm.context(device, 0)
    if not ctx.exist:
        print("Skip because %s is not enabled" % device)
        return
    with tvm.target.create(device):
        dW = topi.nn.dilate(W, (1, 1, dilation, dilation))
        B = topi.nn.conv2d(A, dW, stride, padding, layout='NCHW')
        s1 = topi.generic.schedule_conv2d_nchw([B])
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    b = tvm.nd.array(np.zeros(util.get_const_tuple(B.shape), dtype=B.dtype), ctx)
    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=(device != "cuda")):
        func = tvm.build(s1, [A, W, B], device, name="conv2d_%d_%d_%d_%d_%d_%d_%d_%d" % (batch, in_channel, in_size, num_filter, kernel, stride, padding, dilation))
        #print(tvm.lower(s1, [A, W, B], simple_mode=True))
        func(a, w, b)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        num_runs = 100
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        return timer(a, w, b).mean

def decl_V_minimal(input_tile, alpha, C, P):
    # transform image
    def compute_temp(c, p, i, j):
        now = tvm.const(0.0, "float32")
        temp_expr = {}
        temp_expr[(0,0)] = input_tile[c][p][0][0] - input_tile[c][p][2][0]
        temp_expr[(0,1)] = input_tile[c][p][0][1] - input_tile[c][p][2][1]
        temp_expr[(0,2)] = input_tile[c][p][0][2] - input_tile[c][p][2][2]
        temp_expr[(0,3)] = input_tile[c][p][0][3] - input_tile[c][p][2][3]
        temp_expr[(1,0)] = input_tile[c][p][1][0] + input_tile[c][p][2][0]
        temp_expr[(1,1)] = input_tile[c][p][1][1] + input_tile[c][p][2][1]
        temp_expr[(1,2)] = input_tile[c][p][1][2] + input_tile[c][p][2][2]
        temp_expr[(1,3)] = input_tile[c][p][1][3] + input_tile[c][p][2][3]
        temp_expr[(2,0)] = input_tile[c][p][2][0] - input_tile[c][p][1][0]
        temp_expr[(2,1)] = input_tile[c][p][2][1] - input_tile[c][p][1][1]
        temp_expr[(2,2)] = input_tile[c][p][2][2] - input_tile[c][p][1][2]
        temp_expr[(2,3)] = input_tile[c][p][2][3] - input_tile[c][p][1][3]
        temp_expr[(3,0)] = input_tile[c][p][1][0] - input_tile[c][p][3][0]
        temp_expr[(3,1)] = input_tile[c][p][1][1] - input_tile[c][p][3][1]
        temp_expr[(3,2)] = input_tile[c][p][1][2] - input_tile[c][p][3][2]
        temp_expr[(3,3)] = input_tile[c][p][1][3] - input_tile[c][p][3][3]
        for ii in range(alpha):
            for jj in range(alpha):
                now = tvm.select(tvm.all(i == ii, j == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((C, P, alpha, alpha), compute_temp, name="temp")

    def compute_V(i, j, c, p):
        v_expr = {}
        v_expr[(0, 0)] = temp[c][p][0][0] - temp[c][p][0][2]
        v_expr[(0, 1)] = temp[c][p][0][1] + temp[c][p][0][2]
        v_expr[(0, 2)] = temp[c][p][0][2] - temp[c][p][0][1]
        v_expr[(0, 3)] = temp[c][p][0][1] - temp[c][p][0][3]
        v_expr[(1, 0)] = temp[c][p][1][0] - temp[c][p][1][2]
        v_expr[(1, 1)] = temp[c][p][1][1] + temp[c][p][1][2]
        v_expr[(1, 2)] = temp[c][p][1][2] - temp[c][p][1][1]
        v_expr[(1, 3)] = temp[c][p][1][1] - temp[c][p][1][3]
        v_expr[(2, 0)] = temp[c][p][2][0] - temp[c][p][2][2]
        v_expr[(2, 1)] = temp[c][p][2][1] + temp[c][p][2][2]
        v_expr[(2, 2)] = temp[c][p][2][2] - temp[c][p][2][1]
        v_expr[(2, 3)] = temp[c][p][2][1] - temp[c][p][2][3]
        v_expr[(3, 0)] = temp[c][p][3][0] - temp[c][p][3][2]
        v_expr[(3, 1)] = temp[c][p][3][1] + temp[c][p][3][2]
        v_expr[(3, 2)] = temp[c][p][3][2] - temp[c][p][3][1]
        v_expr[(3, 3)] = temp[c][p][3][1] - temp[c][p][3][3]
        now = tvm.const(0.0, "float32")
        for ii in range(4):
            for jj in range(4):
                now = tvm.select(tvm.all(i == ii, j == jj),
                                 v_expr[(ii, jj)],
                                 now)
        return now

    V = tvm.compute((alpha, alpha, C, P), compute_V)
    
    return V

def decl_output_minimal(M, N, K, H, W, P, m, nH, nW):

    def compute_temp(k, p, eps, nu):
        temp_expr = {}
        for j in range(4):
            t0 = M[0][j][k][p] + M[1][j][k][p]
            t1 = M[1][j][k][p] - M[2][j][k][p]
            temp_expr[(0,j)] = t0 + M[2][j][k][p]
            temp_expr[(1,j)] = t1 - M[3][j][k][p]

        now = tvm.const(0.0, "float32")
        for ii in range(2):
            for jj in range(4):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 temp_expr[(ii, jj)],
                                 now)
        return now

    temp = tvm.compute((K, P, 2,4), compute_temp, name="temp")

    def compute_output(n, k, h, w):
        b = n * nH * nW + (h//m) * nW + w//m
        eps = h%m
        nu = w%m
        output_expr = {}
        for i in range(2):
            t0 = temp[k][b][i][0] + temp[k][b][i][1]
            t1 = temp[k][b][i][1] - temp[k][b][i][2]
            output_expr[(i,0)] = t0 + temp[k][b][i][2]
            output_expr[(i,1)] = t1 - temp[k][b][i][3]

        now = tvm.const(0.0, "float32")
        for ii in range(2):
            for jj in range(2):
                now = tvm.select(tvm.all(eps == ii, nu == jj),
                                 output_expr[(ii, jj)],
                                 now)
        return now

    output = tvm.compute((N, K, H, W), compute_output)

    return output

def decl_winograd(data, U, stride, padding, out_dtype):
    """declare winograd fast convolution F(2x2, 3x3) for conv2d"""
    N, C, H, W = [util.get_const_int(x) for x in data.shape]
    _, _, C, K = [util.get_const_int(x) for x in U.shape]
    HPAD, WPAD = 1,1
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    assert HSTR == 1 and WSTR == 1 and HPAD == 1 and WPAD == 1
    data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")

    m = 2
    r = 3
    alpha = m + r - 1
    K = K
    nH, nW = (H + m-1) // m, (W + m-1) // m
    P = N * nH * nW

    # pack input tile
    input_tile = tvm.compute((C, P, alpha, alpha),
                             lambda c, b, eps, nu:
                             tvm.select(b < P, data_pad[b // (nH*nW)][c][b// nW % nH * m + eps][b % nW * m + nu], tvm.const(0, data_pad.dtype)), name='d')

    V = decl_V_minimal(input_tile, alpha, C, P)

    # batch gemm
    c = tvm.reduce_axis((0, C), name='c')
    M = tvm.compute((alpha, alpha, K, P), lambda eps, nu, k, b:
                    tvm.sum(U[eps][nu][c][k] *
                            V[eps][nu][c][b], axis=c), name='M')

    # inverse transform and unpack
    output = decl_output_minimal(M, N, K, H, W, P, m, nH, nW)

    return output

def schedule_smem_load(s, smem, num_thread):
    yi, xi, ci, ni = s[smem].op.axis
    ty, ci = s[smem].split(ci, nparts=num_thread)
    tx, ni = s[smem].split(ni, nparts=num_thread)
    _, ni = s[smem].split(ni, factor=4)
    s[smem].reorder(ty, tx, yi, xi, ci, ni)
    s[smem].vectorize(ni)  # vectorize memory load
    s[smem].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[smem].bind(tx, tvm.thread_axis("threadIdx.x"))

def schedule_batched_sgemm(s, U, V, M):
    UU = s.cache_read(U, 'shared', [M])
    VV = s.cache_read(V, "shared", [M])
    UL = s.cache_read(UU, "local", [M])
    VL = s.cache_read(VV, "local", [M])
    ML = s.cache_write(M, "local")

    tile = 8
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

    eps, nu, k, p = s[M].op.axis
    ko, ki = s[M].split(k, factor=block_factor)
    po, pi = s[M].split(p, factor=block_factor)
    z = s[M].fuse(eps, nu)

    s[M].bind(z, tvm.thread_axis("blockIdx.z"))
    s[M].bind(ko, tvm.thread_axis("blockIdx.y"))
    s[M].bind(po, tvm.thread_axis("blockIdx.x"))

    tyz, kii = s[M].split(ki, nparts=vthread)  # virtual thread split
    txz, pii = s[M].split(pi, nparts=vthread)  # virtual thread split
    ty, kii = s[M].split(kii, nparts=num_thread)
    tx, pii = s[M].split(pii, nparts=num_thread)
    s[M].reorder(z, ko, po, tyz, txz, ty, tx, kii, pii)

    s[M].bind(tyz, thread_yz)
    s[M].bind(txz, thread_xz)
    s[M].bind(ty, thread_y)
    s[M].bind(tx, thread_x)

    s[ML].compute_at(s[M], tx)
    eps, nu, k, p = s[ML].op.axis
    c = s[ML].op.reduce_axis[0]
    co, ci = s[ML].split(c, factor=step)
    s[ML].reorder(co, ci, k, p)

    s[UU].compute_at(s[ML], co)
    s[VV].compute_at(s[ML], co)
    s[UL].compute_at(s[ML], ci)
    s[VL].compute_at(s[ML], ci)

    schedule_smem_load(s, UU, num_thread)
    schedule_smem_load(s, VV, num_thread)

def schedule_winograd(outs):
    s = tvm.create_schedule([x.op for x in outs])
    op = outs[0].op
    output = op.output(0)
    output_temp = s[output].op.input_tensors[0]
    M = s[output_temp].op.input_tensors[0]
    U, V = s[M].op.input_tensors
    V_temp = s[V].op.input_tensors[0]
    d = s[V_temp].op.input_tensors[0]
    data_pad = s[d].op.input_tensors[0]

    s[data_pad].compute_inline()

    # transform image
    eps, nu, c, p = s[V].op.axis
    s[V].reorder(c, p, eps, nu)

    co, ci = s[V].split(c, factor=16)
    po, pi = s[V].split(p, factor=16)
    s[V].bind(ci, tvm.thread_axis("threadIdx.y"))
    s[V].bind(pi, tvm.thread_axis("threadIdx.x"))
    s[V].bind(co, tvm.thread_axis("blockIdx.y"))
    s[V].bind(po, tvm.thread_axis("blockIdx.x"))
    s[V_temp].compute_at(s[V], pi)
    s[d].compute_at(s[V], pi)
    
    schedule_batched_sgemm(s, U, V, M)

    # inverse transform
    n, k, h, w = s[output].op.axis
    ho, hi = s[output].split(h, factor=2)
    wo, wi = s[output].split(w, factor=2)
    s[output].reorder(k, n, ho, wo, hi, wi)
    k = s[output].fuse(k, n)

    hoo, hoi = s[output].split(ho, factor=16)
    woo, woi = s[output].split(wo, factor=16)
    s[output].reorder(hoo, woo, hoi, woi, hi, wi)
    s[output].bind(hoi, tvm.thread_axis("threadIdx.y"))
    s[output].bind(woi, tvm.thread_axis("threadIdx.x"))
    s[output].bind(hoo, tvm.thread_axis("blockIdx.y"))
    s[output].bind(woo, tvm.thread_axis("blockIdx.x"))
    s[output].bind(k, tvm.thread_axis("blockIdx.z"))
    s[output_temp].compute_at(s[output], woi)

    return s

def transform_filter(w_np):
    num_filter, in_channel, kernel, kernel = w_np.shape
    G = np.array([
        [1, 0, 0],
        [1.0/2, 1.0/2, 1.0/2],
        [1.0/2, -1.0/2, 1.0/2],
        [0, 0, 1],
    ], w_np.dtype)

    out = np.empty((4, 4, in_channel, num_filter), w_np.dtype)
    for i in range(in_channel):
        for j in range(num_filter):
            out[:, :, i, j] = np.dot(G, np.dot(w_np[j, i], G.transpose()))
    return out


def test_winograd(batch, in_channel, in_size, num_filter, kernel, stride, padding, device):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
    W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
    U = tvm.placeholder((4, 4, in_channel, num_filter), name='W')

    a_shape = util.get_const_tuple(A.shape)
    w_shape = util.get_const_tuple(W.shape)
    dtype = A.dtype
    dilation = 1

    @memoize("topi.tests.test_topi_conv2d_nchw.wino")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1, 1, dilation, dilation))
        b_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
        c_np = np.maximum(b_np, 0)
        return a_np, w_np, b_np, c_np

    a_np, w_np, b_np, c_np = get_ref_data()

    with tvm.target.create(device):
        B = decl_winograd(A, U, stride, padding, dtype)
        s = schedule_winograd([B])

    u_np = transform_filter(w_np)

    ctx = tvm.context(device, 0)
    a = tvm.nd.array(a_np, ctx)
    u = tvm.nd.array(u_np, ctx)
    b = tvm.nd.array(np.zeros(util.get_const_tuple(B.shape), dtype=B.dtype), ctx)
    with tvm.build_config(auto_unroll_max_step=1400,
                          unroll_explicit=(device != "cuda"),
                          partition_const_loop=False):
        func = tvm.build(s, [A, U, B], device)
        #print(tvm.lower(s, [A, U, B], simple_mode=True))
        func(a, u, b)
        num_runs = 100
        timer = func.time_evaluator(func.entry_name, ctx, number=num_runs)
        np.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)
        #print(func.imported_modules[0].get_source())
        return timer(a, u, b).mean

# for copy paste as markdown
def generate_table(workloads, wino_times, direct_times, wino_nvptx_times, direct_nvptx_times, lib_times, lib_name):
    print("| (batch,CI,size,CO) | TVM Winograd (This code) | TVM Direct | TVM Winograd NVPTX (This code) | TVM Direct NVPTX | %s |" % lib_name)
    print("|------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|")
    for (workload, t_wino, t_direct, t_wino_nvptx, t_direct_nvptx, t_lib) in zip(workloads, wino_times, direct_times, wino_nvptx_times, direct_nvptx_times, lib_times):
        if t_direct and t_direct_nvptx:
            print("|", workload, "| %.3f | %.3f | %.3f | %.3f | %.3f" % (t_wino,  t_direct, t_wino_nvptx, t_direct_nvptx, t_lib))
        elif t_direct:
            print("|", workload, "| %.3f | %.3f | %.3f | N/A | %.3f" % (t_wino,  t_direct, t_wino_nvptx, t_lib))
        elif t_direct_nvptx:
            print("|", workload, "| %.3f | N/A | %.3f | %.3f | %.3f" % (t_wino, t_wino_nvptx, t_direct_nvptx, t_lib))
        else:
            print("|", workload, "| %.3f | N/A | %.3f | N/A | %.3f" % (t_wino, t_wino_nvptx, t_lib))


workloads = [(1, 128, 122, 128),
             (1, 128, 128, 128),
             (1, 64, 56, 64),
             (1, 64, 64, 32),
             (1, 64, 224, 64),
             (1, 64, 112, 128),
             (1, 512, 28, 512),
             (1, 128, 28, 128),
             (1, 256, 14, 256),
             (8, 128, 122, 128),
             (16, 64, 56, 64),
             (32, 64, 64, 32),
             (64, 128, 32, 128)
            ]

vgg_workloads = [(1, 64, 224, 64), #relu, input and output transform slow
                 (1, 64, 112, 128),#relu2
                 (1, 128, 112, 128),
                 (1, 128, 56, 256),
                 (1, 256, 56, 256), #relu4
                 (1, 256, 28, 512),
                 (1, 512, 28, 512), # relu6
                 (1, 512, 14, 512) # relu7
                 ]

wino_times = []
direct_times = []
wino_nvptx_times = []
direct_nvptx_times = []
lib_times = []
device = "cuda"

for workload in workloads:
    t_wino = test_winograd(*workload, 3, 1, 1, device)
    t_wino_nvptx = test_winograd(*workload, 3, 1, 1, "nvptx")

    if workload[1] == 512 or workload[0] > 1:
        t_direct = None # tvm direct conv2d cannot handle this workload
        t_direct_nvptx = None
    else:
        t_direct = reference_direct(*workload, 3, 1, 1, device)
        if workload[2] == 122:
            t_direct_nvptx = None
        else:
            t_direct_nvptx = reference_direct(*workload, 3, 1, 1, "nvptx")

    #t_lib = reference_direct(*workload, 3, 1, 1, "cuda -libs=cudnn")
    t_lib = 0
    wino_times.append(t_wino * 1000)
    wino_nvptx_times.append(t_wino_nvptx * 1000)
    lib_times.append(t_lib * 1000)

    if t_direct:
        t_direct *= 1000
    if t_direct_nvptx:
        t_direct_nvptx *= 1000

    direct_times.append(t_direct)
    direct_nvptx_times.append(t_direct_nvptx)

generate_table(workloads, wino_times, direct_times, wino_nvptx_times, direct_nvptx_times, lib_times, "cuDNN Winograd")


