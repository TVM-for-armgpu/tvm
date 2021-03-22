import tvm
from tvm import te, topi
from tvm.topi import nn

# Algorithm
# kernel transform
def kernel_transform(kernel, wino_size, G=None, out_dtype='float32'):
    if isinstance(kernel, tuple):
        kernel_shape = kernel
        kernel_dtype = 'float32'
        filter_n = te.placeholder(
            kernel_shape, name="filter", dtype=kernel_dtype)
    else:
        kernel_shape = tuple(int(i) for i in kernel.shape)
        filter_n = kernel


    if out_dtype == "climgfloatw32":
        storage_attr = "image"
    else:
        storage_attr = ""
    assert len(kernel_shape) == 6
    IC, oc_chunk, KH, KW, _, oc_bn = kernel_shape

    k1 = te.reduce_axis((0, KH), "k1")
    k2 = te.reduce_axis((0, KW), "k2")
    G = G if G is not None else te.placeholder((wino_size, KH), name="G")

    U = te.compute((IC, oc_chunk, wino_size, wino_size, 1, oc_bn), lambda ic, oc, x, y, bi, bo:
                   te.sum(G[x, k2] * filter_n[ic, oc, k2, k1, bi, bo]*G[y, k1],
                          axis=[k2, k1]),
                   name="mali_conv2d_nchw_winograd_U",
                   attrs={"data_type": storage_attr},
                   )
    return U

# Default schedule
#s = te.create_schedule(U.op)
#========shedule begin=================================
def kernel_transform_shedule(cfg, s, U):
    G, filter_n = s[U].op.input_tensors
    if isinstance(filter_n.op, tvm.te.tensor.ComputeOp):
        s[filter_n].compute_inline()
    s[G].compute_inline()
    WL = s.cache_read(filter_n, "local", [U])
    UL = s.cache_write(U, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    #block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    #thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    cp, kp, Uhp, Uwp, _, Up4 = s[U].op.axis
    cpo, cpi = cfg["kernel_cp"].apply(s, U, cp)
    kpo, kpi = cfg["kernel_kp"].apply(s, U, kp)

    #cpo, cpi = s[U].split(cp, factor=4)
    #kpo, kpi = s[U].split(kp, factor=4)

    s[U].bind(kpo, block_x)
    s[U].bind(cpo, block_y)
    s[U].bind(kpi, thread_x)
    s[U].bind(cpi, thread_y)
    s[U].reorder(cpo, cpi, kpi, Uhp, Uwp, Up4, kpo)

    # Schedule BL local write
    s[UL].compute_at(s[U], kpo)
    _, _, hp, wp, _, p4 = s[UL].op.axis
    s[UL].reorder(hp, wp, p4)
    s[UL].vectorize(p4)
    k1, k2 = s[UL].op.reduce_axis
    #s[UL].unroll(wp)
    #s[UL].unroll(hp)
    #s[UL].unroll(k1)
    #s[UL].unroll(k2)

    s[WL].compute_at(s[UL], hp)
    _, _, hp, wp, _, p4 = s[WL].op.axis
    s[WL].vectorize(p4)  # vectorize memory load
    #s[WL].unroll(wp)
    #s[WL].unroll(hp)
    s[U].vectorize(Up4)  # vectorize memory load
    #s[U].unroll(Uwp)  # vectorize memory load
    #s[U].unroll(Uhp)  # vectorize memory load
#========shedule end=================================
#func = tvm.build(s, [filter_n,U], target=target)
#assert func
#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")

#print(tvm.lower(s, [filter_n,U], simple_mode=True))

#c = tvm.nd.array(numpy.zeros((M, M), dtype=dtype), ctx)
#func(a, b, c)
#tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
#
#evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
#print("Baseline: %f" % evaluator(a, b, c).mean)

# data transform


def data_transform(data, wino_size, padding, B=None, out_dtype='float32'):
    if isinstance(data, tuple):
        data_shape = data
        data_dtype = 'float32'
        Data = te.placeholder(data_shape,
                          name="Data", dtype=data_dtype)
    else:
        data_shape = tuple(int(i) for i in data.shape)
        Data = data

    if out_dtype == "climgfloatw32":
        storage_attr = "image"
    else:
        storage_attr = ""
    assert len(data_shape) == 5
    pt, pl, pb, pr=padding
    N, ic_chunck, img_H, img_W, ic_bn = data_shape
    H = (img_H + pt + pb - 3) // 1 + 1
    W = (img_W + pl + pr - 3) // 1 + 1
    tile_size = wino_size - 2
    wino2H = (H + tile_size - 1) // tile_size
    wino2W = (W + tile_size - 1) // tile_size

    k1 = te.reduce_axis((0, wino_size), "r_a")
    k2 = te.reduce_axis((0, wino_size), "r_b")
    B = B if B is not None else te.placeholder((wino_size, wino_size), name="B")

    #tx=(int(y / w) * wino_size + x / wino_size);
    #ty = (y % w) * wino_size + x % wino_size;
    #ox = int(tx / wino_size) * 2+ k2;
    #oy = int(ty / wino_size * 2 + k1;
    #V[x][y] += BT(tx % wino_size, k2) * D[ox ][oy] * B(k1, (ty % wino_size);
    Data_pad = nn.pad(Data, (0, 0, pt, pl, 0), (0, 0, pb, pr, 0),
                      name="Data_pad")

    def _transform_V(ic, oc, x, y, bo):
        tx = y // wino2W * wino_size + x // wino_size
        ty = (y % wino2W) * wino_size + x % wino_size
        ox = tx // wino_size * tile_size + k2 - 1
        oy = ty // wino_size * tile_size + k1 - 1
        return te.sum(B[k2, tx % wino_size]*Data_pad[ic, oc, ox, oy, bo]*B[k1, ty % wino_size],
                      axis=[k2, k1],
                      )
    V = te.compute((N, ic_chunck, wino_size*wino_size, wino2H * wino2W, ic_bn),
                   _transform_V,
                   name="mali_conv2d_nchw_winograd_V",
                   attrs={"data_type": storage_attr},
                   )
    return V
#s = te.create_schedule(V.op)
##========shedule begin=================================


def data_transform_shedule(cfg,  s, V):
    B, Data_pad = s[V].op.input_tensors
    s[B].compute_inline()
    s[Data_pad].compute_inline()

    if isinstance(Data_pad.op, tvm.te.tensor.ComputeOp):
        packed_data, = s[Data_pad].op.input_tensors
        if isinstance(packed_data.op, tvm.te.tensor.ComputeOp):
            s[packed_data].compute_inline()
    WL = s.cache_read(Data_pad, "local", [V])
    VL = s.cache_write(V, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    n, cp, hp, wp, Vp4 = s[V].op.axis

    cpo, cpi = cfg["data_cp"].apply(s, V, cp)
    wpo, wpi, Vwp = cfg["data_wp"].apply(s, V, wp)
    hpo, hpi, Vhp = cfg["data_hp"].apply(s, V, hp)

    #cpo, cpi = s[V].split(cp, factor=4)

    #wp, Vwp = s[V].split(wp, factor=4)
    #wpo, wpi = s[V].split(wp, factor=4)

    #hp, Vhp = s[V].split(hp, factor=4)
    #hpo, hpi = s[V].split(hp, factor=4)

    s[V].bind(wpo, block_x)
    s[V].bind(hpo, block_y)
    s[V].bind(cpo, block_z)
    s[V].bind(wpi, thread_x)
    s[V].bind(hpi, thread_y)
    s[V].bind(cpi, thread_z)
    s[V].reorder(cpo, cpi, hpo, hpi, wpi, wpo, Vp4, n)

    # Schedule BL local write
    s[VL].compute_at(s[V], n)
    _, _, hp, wp, p4 = s[VL].op.axis
    s[VL].reorder(hp, wp, p4)
    s[VL].vectorize(p4)
    k1, k2 = s[VL].op.reduce_axis
    #s[VL].unroll(wp)
    #s[VL].unroll(hp)
    #s[VL].unroll(k1)
    #s[VL].unroll(k2)

    s[WL].compute_at(s[VL], hp)
    _, _, hp, wp, p4 = s[WL].op.axis
    s[WL].vectorize(p4)  # vectorize memory load
    #s[WL].unroll(wp)
    #s[WL].unroll(hp)
    s[V].vectorize(Vp4)  # vectorize memory load
    #s[V].unroll(Vwp)  # vectorize memory load
    #s[V].unroll(Vhp)  # vectorize memory load
#========shedule end=================================
#func = tvm.build(s, [Data,V], target=target)
#assert func
#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
##print(tvm.lower(s, [Data,V], simple_mode=True))
#exit(0)


def batch_gemm(U=None, V=None, out_dtype='float32'):
    if out_dtype == "climgfloatw32":
        storage_attr = "image"
    else:
        storage_attr = ""
    U = U if U is not None else te.placeholder((256, 256//4, 4, 4,
                        1, 4), name="U")
    V = V if V is not None else te.placeholder((1, 256//4, 16, 3 * 5, 4), name="V")
    IC, oc_chunk, wino_size, _, _, oc_bn = U.shape
    N, _, wino_all_size, NTILE, _ = V.shape
    assert V.shape[1] * V.shape[-1] == U.shape[
        0], f'CIn channel must the same {V.shape} vs {U.shape}'
    rc = te.reduce_axis((0, IC), "r_c")
    M = te.compute((N, oc_chunk, wino_all_size, NTILE, oc_bn), lambda n, oc, h, w, bo:
                   te.sum(U[rc, oc, h//wino_size, h % wino_size, 0, bo]*V[n, rc//oc_bn, h, w, rc % oc_bn],
                          axis=[rc]),
                   name="mali_conv2d_nchw_winograd_M",
                   attrs={"data_type": storage_attr},
                   )
    return M
#s = te.create_schedule(M.op)

##========shedule begin=================================


def batch_gemm_shedule(cfg, s, M):
    U, V = s[M].op.input_tensors
    UL = s.cache_read(U, "local", [M])
    VL = s.cache_read(V, "local", [M])
    ML = s.cache_write(M, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    n, kp, hp, wp, Mp4 = s[M].op.axis

    kpo, kpi = cfg["bgemm_kp"].apply(s, M, kp)
    wpo, wpi, Mwp = cfg["bgemm_wp"].apply(s, M, wp)
    hpo, hpi = cfg["bgemm_hp"].apply(s, M, hp)

    #kpo, kpi = s[M].split(kp, factor=4)
    #wp, Mwp = s[M].split(wp, factor=4)
    #wpo, wpi = s[M].split(wp, factor=4)
    #hpo, hpi = s[M].split(hp, factor=4)

    s[M].bind(wpo, block_x)
    s[M].bind(hpo, block_z)
    s[M].bind(kpo, block_y)
    s[M].bind(wpi, thread_x)
    s[M].bind(hpi, thread_z)
    s[M].bind(kpi, thread_y)
    s[M].reorder(kpo, hpo, hpi, wpi, wpo, Mwp, Mp4, kpi)

    # Schedule BL local write
    s[ML].compute_at(s[M], kpi)
    s[M].reorder(kpi, Mwp, Mp4)

    _, _, hp, wp, p4 = s[ML].op.axis
    rc, = s[ML].op.reduce_axis
    rco, rci = s[ML].split(rc, factor=4)
    s[ML].reorder(rco, wp, rci, p4)
    s[ML].vectorize(p4)
    #s[ML].unroll(wp)
    #s[ML].unroll(hp)
    #s[ML].unroll(k1)

    #schedule UL VL local read
    s[UL].compute_at(s[ML], rco)
    s[VL].compute_at(s[ML], rco)

    #split Ul VL workload
    a, b, hp, wp, _, p4 = s[UL].op.axis
    s[UL].vectorize(p4)  # vectorize memory load
    #s[UL].unroll(wp)
    #s[UL].unroll(hp)
    _, _, hp, wp, p4 = s[VL].op.axis
    s[VL].vectorize(p4)  # vectorize memory load
    #s[VL].unroll(wp)
    #s[VL].unroll(hp)

    s[M].vectorize(Mp4)  # vectorize memory load
    #s[M].unroll(Mwp)  # vectorize memory load
    #s[M].unroll(Mhp)  # vectorize memory load
#========shedule end=================================

#func = tvm.build(s, [U,V,M], target=target)
#assert func
#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
#print(tvm.lower(s, [U,V,M], simple_mode=True))
#print(tvm.lower(s, [filter_n,Data,M], simple_mode=True))

# inverse transform


def inverse_transform(out_shape, A=None, M=None, out_dtype='float32'):
    if out_dtype == "climgfloatw32":
        storage_attr = "image"
    else:
        storage_attr = ""
    A = A if A is not None else te.placeholder((4, 2), name="A")
    M = M if M is not None else te.placeholder((1, 256 // 4, 16,
                        3 * 5, 4), name="M")
    N, oc_chunk, wino_all_size, NTILE, oc_bn = M.shape
    _, _, img_H, img_W, _ = out_shape
    tile_size = A.shape[1]
    wino2H = (img_H + tile_size - 1) // tile_size
    wino2W = (img_W + tile_size - 1) // tile_size
    wino_size = tile_size + 2

    assert wino_size * wino_size == wino_all_size
    assert wino2H * wino2W == NTILE
    assert out_shape[1] == oc_chunk
    k1 = te.reduce_axis((0, wino_size), "r_a")
    k2 = te.reduce_axis((0, wino_size), "r_b")

    def _transform_inverse(n, oc, h, w, bo):
        th = h // tile_size * wino_size + k2
        tw = w // tile_size * wino_size + k1
        oh = (th % wino_size + tw // wino_size) * \
            wino_size + tw % wino_size
        ow = tw // wino_size + (th // wino_size) * wino2W
        #AT(th % tile_size, k2) * M[ox][oy] * A(k1, tw % 2)
        return te.sum(A[k2, th % tile_size]*M[n, oc, oh, ow, bo]*A[k1, tw % tile_size],
                      axis=[k1, k2])

    #int th = int(h / tile_size) * wino_size + k2;
    #int tw = int(w / tile_size) * wino_size + k1;
    #
    #int ox = (th % wino_size) * wino_size + tw % wino_size;
    #int oy = int(tw / wino_size) + int(th / wino_size) * trans_image_size_block_W;
    #output[h][w] += AT(th % tile_size, k2) * M[ox][oy] * A(k1, tw % 2);

    output = te.compute((N, oc_chunk, img_H, img_W, oc_bn), _transform_inverse,
                        name="mali_conv2d_nchw_winograd_output",
                        attrs={"data_type": storage_attr},
                        tag="winograd_conv2d_output",
                        )
    return output

#s = te.create_schedule(output.op)


def inverse_transform_shedule(cfg, s, op):
    ##========shedule begin=================================
    output = op.output(0)
    A, M = s[output].op.input_tensors
    s[A].compute_inline()
    ML = s.cache_read(M, "local", [output])
    OL = s.cache_write(output, "local")
    #fuse-==========
    if op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)
    O = output
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    len_4_flag = False
    if len(s[O].op.axis) == 4:
        len_4_flag = True
        n, cp, hp, wp = s[O].op.axis
    else:
        n, cp, hp, wp, Op4 = s[O].op.axis

    cpo, cpi = cfg["inv_cp"].apply(s, O, cp)
    wpo, wpi, Owp=cfg["inv_wp"].apply(s, O, wp)
    hpo, hpi, Ohp=cfg["inv_hp"].apply(s, O, hp)

    #cpo, cpi = s[O].split(cp, factor=4)
    #wp, Owp = s[O].split(wp, factor=4)
    #wpo, wpi = s[O].split(wp, factor=4)
    #hp, Ohp = s[O].split(hp, factor=4)
    #hpo, hpi = s[O].split(hp, factor=4)

    if len_4_flag:
        cpi, Op4 = s[O].split(cpi, factor=4)

    s[O].bind(wpo, block_x)
    s[O].bind(hpo, block_y)
    s[O].bind(cpo, block_z)
    s[O].bind(wpi, thread_x)
    s[O].bind(hpi, thread_y)
    s[O].bind(cpi, thread_z)
    s[O].reorder(cpo, cpi, hpo, hpi, wpi, wpo, Owp, Ohp, Op4, n)

    # Schedule BL local write
    s[OL].compute_at(s[O], n)
    _, _, hp, wp, p4 = s[OL].op.axis
    s[OL].reorder(hp, wp, p4)
    s[OL].vectorize(p4)
    k1, k2 = s[OL].op.reduce_axis
    #s[OL].unroll(wp)
    #s[OL].unroll(hp)
    #s[OL].unroll(k1)
    #s[OL].unroll(k2)

    s[ML].compute_at(s[OL], hp)
    _, _, hp, wp, p4 = s[ML].op.axis
    s[ML].vectorize(p4)  # vectorize memory load
    #s[ML].unroll(wp)
    #s[ML].unroll(hp)
    s[O].vectorize(Op4)  # vectorize memory load

    #s[O].unroll(Owp)  # vectorize memory load
    #s[O].unroll(Ohp)  # vectorize memory load
#========shedule end=================================


def _schedule_winograd_nchwc_io(cfg, s, op):
    output = op.output(0)
    A, M = s[output].op.input_tensors
    U, V = s[M].op.input_tensors
    G, filter_n = s[U].op.input_tensors
    B, DataPad = s[V].op.input_tensors
    Data, = s[DataPad].op.input_tensors
    #==============
    inverse_transform_shedule(cfg, s, op)
    U, V = s[M].op.input_tensors
    batch_gemm_shedule(cfg, s, M)
    kernel_transform_shedule(cfg, s, U)
    data_transform_shedule(cfg, s, V)
#========shedule end=================================

#func = tvm.build(s, [M,output], target=target, name="mmult")
##assert func
##print(tvm.lower(s, [M,output], simple_mode=True))
##print(tvm.lower(s, [filter_n,Data,output], simple_mode=True))
#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
#exit(0)
