import tvm
from tvm import te, topi
from tvm.topi import nn
import tvm.testing
from tvm import te
import numpy
import numpy as np


N = 1
IC = 512
OC = 512
img_H = 4 * 16
img_W = 4 * 8

dtype = "float32"

K = 3
r = K


tile_size = 2
wino_size = tile_size + 2
wino_all_size = wino_size*wino_size

wino2H = (img_H+tile_size-1)//tile_size
wino2W = (img_W+tile_size-1)//tile_size
H_DIV2 = wino2H
W_DIV2 = wino2W
NTILE = wino2H * wino2H
# Algorithm
# kernel transform
def kernel_transform(G=None):
    k1 = te.reduce_axis((0, K), "k1")
    k2 = te.reduce_axis((0, K), "k2")
    G = te.placeholder((wino_size, K), name="G") if G==None else G
    filter_n = te.placeholder((IC,OC//4,K, K,1,4), name="filter")
    U = te.compute((IC, OC//4, wino_size, wino_size, 1, 4), lambda ic, oc, x, y, bi, bo:
                  te.sum(G[x, k2] * filter_n[ic, oc, k2, k1, bi, bo]*G[y, k1],
                         axis=[k2, k1]),
                  name="U")
    return U
# Default schedule
#s = te.create_schedule(U.op)
#========shedule begin=================================


def kernel_transform_shedule(s, U):
    G, filter_n = s[U].op.input_tensors
    s[G].compute_inline()
    WL = s.cache_read(filter_n, "local", [U])
    UL = s.cache_write(U, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis( "threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    cp, kp, Uhp, Uwp, _, Up4  = s[U].op.axis
    cpo,cpi=s[U].split(cp,factor=4)
    kpo,kpi=s[U].split(kp,factor=4)
    s[U].bind(cpo, block_x)
    s[U].bind(kpo, block_y)
    s[U].bind(kpi, thread_x)
    s[U].bind(cpi, thread_y)
    s[U].reorder(cpo,cpi,kpi,Uhp,Uwp,Up4,kpo)

    # Schedule BL local write
    s[UL].compute_at(s[U],kpo)
    _,_,hp, wp, _, p4  = s[UL].op.axis
    s[UL].reorder(hp,wp,p4)
    s[UL].vectorize(p4)
    k1,k2 = s[UL].op.reduce_axis
    #s[UL].unroll(wp)
    #s[UL].unroll(hp)
    #s[UL].unroll(k1)
    #s[UL].unroll(k2)

    s[WL].compute_at(s[UL], hp)
    _,_,hp, wp,_, p4 = s[WL].op.axis
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
def data_transform(B=None):
    k1 = te.reduce_axis((0, wino_size), "r_a")
    k2 = te.reduce_axis((0, wino_size), "r_b")
    B = te.placeholder((wino_size, wino_size), name="B") if B==None else B
    Data = te.placeholder((1,IC//4,img_H, img_W,4), name="Data")
    N=1
    #tx=(int(y / w) * wino_size + x / wino_size);
    #ty = (y % w) * wino_size + x % wino_size;
    #ox = int(tx / wino_size) * 2+ k2;
    #oy = int(ty / wino_size * 2 + k1;
    #V[x][y] += BT(tx % wino_size, k2) * D[ox ][oy] * B(k1, (ty % wino_size);    
    pt = pl = pb = pr = tile_size - 1
    Data_pad = nn.pad(Data, (0, 0, pt, pl, 0), (0, 0, pb, pr, 0), name="Data_pad")    
    def _transform_V(ic, oc, x, y, bo):
           tx = y // wino2W * wino_size + x // wino_size
           ty = (y % wino2W) * wino_size + x % wino_size
           ox = tx // wino_size * tile_size + k2 - 1
           oy = ty // wino_size * tile_size + k1 - 1
           return te.sum(B[k2, tx % wino_size]*Data_pad[ic, oc, ox, oy, bo]*B[k1, ty % wino_size],
                         axis=[k2, k1],
                         )    
    V = te.compute((N, IC // 4, wino_size*wino_size, wino2H * wino2W, 4),
                  _transform_V,
                  name="V")
    return V
#s = te.create_schedule(V.op)
##========shedule begin=================================


def data_transform_shedule(s, V):
    B, Data_pad = s[V].op.input_tensors
    s[B].compute_inline()
    s[Data_pad].compute_inline()
    WL = s.cache_read(Data_pad, "local", [V])
    VL = s.cache_write(V, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis( "threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    n, cp, hp, wp, Vp4  = s[V].op.axis
    cpo,cpi=s[V].split(cp,factor=4)
    wp,Vwp=s[V].split(wp,factor=4)
    wpo,wpi=s[V].split(wp,factor=4)
    hp,Vhp=s[V].split(hp,factor=4)
    hpo,hpi=s[V].split(hp,factor=4)
    s[V].bind(wpo, block_x)
    s[V].bind(hpo, block_y)
    s[V].bind(cpo, block_z)
    s[V].bind(wpi, thread_x)
    s[V].bind(hpi, thread_y)
    s[V].bind(cpi, thread_z)
    s[V].reorder(cpo,cpi,hpo,hpi,wpi,wpo,Vp4,n)

    # Schedule BL local write
    s[VL].compute_at(s[V],n)
    _,_, hp, wp, p4  = s[VL].op.axis
    s[VL].reorder(hp,wp,p4)
    s[VL].vectorize(p4)
    k1,k2 = s[VL].op.reduce_axis
    #s[VL].unroll(wp)
    #s[VL].unroll(hp)
    #s[VL].unroll(k1)
    #s[VL].unroll(k2)

    s[WL].compute_at(s[VL], hp)
    _,_,hp, wp, p4 = s[WL].op.axis
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


ddtype = "climgfloatr32"
def batch_gemm(U=None,V=None):
    rc = te.reduce_axis((0, IC), "r_c")
    U = te.placeholder((IC, OC//4, wino_size, wino_size, 1, 4),
                       name="U", dtype=ddtype) if U == None else U
    V = te.placeholder((N, IC//4, wino_all_size, (wino2H)
                        * (wino2W), 4), name="V", dtype=ddtype) if V == None else V
    M = te.compute((N, OC//4, wino_all_size, wino2H * wino2W, 4), lambda n, oc, h, w, bo:
                   te.sum(U[rc, oc, h//wino_size, h % wino_size, 0, bo]*V[n, rc//4, h, w, rc % 4],
                          axis=[rc]),
                   name="M"
                   )
    M.dtype = ddtype.replace('r','w')
    return U,V,M


##========shedule begin=================================


def batch_gemm_shedule(s, M):
    U, V = s[M].op.input_tensors
    UL = s.cache_read(U, "local", [M])
    VL = s.cache_read(V, "local", [M])
    ML = s.cache_write(M, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis( "threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    n, kp, hp, wp, Mp4  = s[M].op.axis
    kpo,kpi=s[M].split(kp,factor=4)
    wp,Mwp=s[M].split(wp,factor=4)
    wpo,wpi=s[M].split(wp,factor=4)
    hpo,hpi=s[M].split(hp,factor=4)
    s[M].bind(wpo, block_x)
    s[M].bind(hpo, block_z)
    s[M].bind(kpo, block_y)
    s[M].bind(wpi, thread_x)
    s[M].bind(hpi, thread_z)
    s[M].bind(kpi, thread_y)
    s[M].reorder(kpo,hpo,hpi,wpi,wpo,Mwp,Mp4,kpi)

    # Schedule BL local write
    s[ML].compute_at(s[M],kpi)
    s[M].reorder(kpi,Mwp,Mp4)

    _,_, hp, wp, p4  = s[ML].op.axis
    rc, = s[ML].op.reduce_axis
    rco,rci = s[ML].split(rc,factor=4)
    s[ML].reorder(rco,wp,rci,p4)
    s[ML].vectorize(p4)
    #s[ML].unroll(wp)
    #s[ML].unroll(hp)
    #s[ML].unroll(k1)


    #schedule UL VL local read
    s[UL].compute_at(s[ML], rco)
    s[VL].compute_at(s[ML], rco)

    #split Ul VL workload
    a,b,hp, wp, _,p4 = s[UL].op.axis
    s[UL].vectorize(p4)  # vectorize memory load
    #s[UL].unroll(wp)
    #s[UL].unroll(hp)
    _,_,hp, wp, p4 = s[VL].op.axis
    s[VL].vectorize(p4)  # vectorize memory load
    #s[VL].unroll(wp)
    #s[VL].unroll(hp)

    s[M].vectorize(Mp4)  # vectorize memory load


    #s[M].unroll(Mwp)  # vectorize memory load
    #s[M].unroll(Mhp)  # vectorize memory load
#========shedule end=================================

#U, V, M = batch_gemm()
#s = te.create_schedule(M.op)
#batch_gemm_shedule(s,M)
#func = tvm.build(s, [U,V,M], target="opencl")
#assert func
#dsp = tuple(int(i) for i in U.shape)
#fsp = tuple(int(i) for i in V.shape)
#osp = tuple(int(i) for i in M.shape)
#target = "opencl"
#ctx = tvm.context(target, 0)
#data_tvm = tvm.nd.array(numpy.random.rand(*dsp).astype('float32'), ctx,dtype=ddtype)
#filter_tvm = tvm.nd.array(numpy.random.rand(
#    *fsp).astype('float32'), ctx, dtype=ddtype)
#output_tvm = tvm.nd.empty(osp, ctx=ctx, dtype=ddtype.replace('r', 'w'))
#evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
#gflops = numpy.prod(np.array(osp[2:4] + fsp).astype('float')) / 1e9 * 2
#t_cost = evaluator(data_tvm, filter_tvm, output_tvm).mean

#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
#print(tvm.lower(s, [U,V,M], simple_mode=True))
#print(tvm.lower(s, [filter_n,Data,M], simple_mode=True))
exit(0)

def inverse_transform(A=None,M=None):
    # inverse transform
    k1 = te.reduce_axis((0, wino_size), "r_a")
    k2 = te.reduce_axis((0, wino_size), "r_b")
    A = te.placeholder((4, 2), name="A") if A == None else A
    M = te.placeholder((N, OC // 4, wino_all_size,
                          wino2H * wino2W, 4), name="M") if M == None else M

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

    output = te.compute((1, OC//4, img_H, img_W, 4), _transform_inverse,
                        name="output"
                        )
    return output

#s = te.create_schedule(output.op)


def inverse_transform_shedule(s, output):
##========shedule begin=================================
    O = output
    A, M = s[O].op.input_tensors
    s[A].compute_inline()
    ML = s.cache_read(M, "local", [O])
    OL = s.cache_write(O, "local")
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    # Split the workloads
    n, cp, hp, wp, Op4  = s[O].op.axis
    cpo,cpi=s[O].split(cp,factor=4)

    wp,Owp=s[O].split(wp,factor=4)
    wpo,wpi=s[O].split(wp,factor=4)

    hp,Ohp=s[O].split(hp,factor=4)
    hpo,hpi=s[O].split(hp,factor=4)

    s[O].bind(wpo, block_x)
    s[O].bind(hpo, block_y)
    s[O].bind(cpo, block_z)
    s[O].bind(wpi, thread_x)
    s[O].bind(hpi, thread_y)
    s[O].bind(cpi, thread_z)
    s[O].reorder(cpo,cpi,hpo,hpi,wpi,wpo,Owp,Ohp,Op4,n)

    # Schedule BL local write
    s[OL].compute_at(s[O],n)
    _,_, hp, wp, p4  = s[OL].op.axis
    s[OL].reorder(hp,wp,p4)
    s[OL].vectorize(p4)
    k1,k2 = s[OL].op.reduce_axis
    #s[OL].unroll(wp)
    #s[OL].unroll(hp)
    #s[OL].unroll(k1)
    #s[OL].unroll(k2)

    s[ML].compute_at(s[OL], hp)
    _,_,hp, wp, p4 = s[ML].op.axis
    s[ML].vectorize(p4)  # vectorize memory load
    #s[ML].unroll(wp)
    #s[ML].unroll(hp)
    s[O].vectorize(Op4)  # vectorize memory load


    #s[O].unroll(Owp)  # vectorize memory load
    #s[O].unroll(Ohp)  # vectorize memory load
#========shedule end=================================
def _schedule_winograd_nchwc_io(s, output):
    O = output
    A, M = s[O].op.input_tensors
    U, V = s[M].op.input_tensors
    G, filter_n = s[U].op.input_tensors
    B, DataPad = s[V].op.input_tensors
    Data, = s[DataPad].op.input_tensors
    #========
    inverse_transform_shedule(s, output)
    batch_gemm_shedule(s, M)
    kernel_transform_shedule(s, U)
    data_transform_shedule(s, V)
#========shedule end=================================

#func = tvm.build(s, [M,output], target=target, name="mmult")
##assert func
##print(tvm.lower(s, [M,output], simple_mode=True))
##print(tvm.lower(s, [filter_n,Data,output], simple_mode=True))
#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
#exit(0)
