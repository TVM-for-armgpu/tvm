import numpy as np
import tvm
from tvm import te, topi, autotvm
import os
from tvm import te
import sys
sys.path.append('..')

print(os.getpid())
# The sizes of inputs and filters
batch = 1
in_channel = 256
out_channel = 512
in_size = 128
kernel = 1
pad = 0
stride = 1

open_image=1
ddtype='float32'
if open_image == 1:
    ddtype = "climgfloatr32"

# Algorithm
PACK4 = 4
#PACK2=2
PACK2=1
W_P = (in_size+3)//4*4//PACK2
H_P = (in_size+1)//2*2//PACK2
C_P = (in_channel+3)//PACK4
K_P=out_channel//PACK4

#A = te.placeholder((C_P,H_P,PACK2,W_P,PACK2,PACK4), dtype=ddtype,name="A")
A = te.placeholder((1, C_P, H_P, W_P, PACK4), dtype=ddtype,name="A")
W = te.placeholder((1,C_P,3,3,1,PACK4), dtype=ddtype, name="W")
out_size = in_size
# Pad input
Apad = A
# Create reduction variables
rw = te.reduce_axis((0, C_P*PACK4), name="kw")
rh = te.reduce_axis((0, C_P*PACK4), name="kh")
# Compute the convolution
idxdiv = tvm.tir.indexdiv
idxmod = tvm.tir.indexmod
#B = te.compute(
#    (K_P, H_P,W_P*PACK4),
#    #lambda ff,yy,xx_p4: extern_op.mysum(
#    lambda ff,yy,xx_p4: te.sum(
#        #extern_op.mymul(Apad[rc//4,yy,rc%4+xx_p4//4*4] , W[rc,ff*PACK4+idxmod(xx_p4,4)]), axis=[rc]
#        Apad[rc//4,yy,rc%4+xx_p4//4*4] * W[rc,ff*PACK4+idxmod(xx_p4,4)], axis=[rc]
#    ),
#    name="B",
#)
kh=kw=3
oc_bn =ic_bn= 4
channel_multiplier = 1
HSTR=WSTR = 1
dh = dw = 0
pad_before = [0, 0, 1, 1, 0]
pad_after = [0, 0, 1, 1, 0]
Apad = tvm.topi.nn.pad(A, pad_before, pad_after, name="PaddedInput")
B = te.compute(
        (1, C_P, H_P, W_P, 4),
        lambda b, occk, oh, ow, ocb: te.sum(
            (
                Apad[
                    b,
                    idxdiv(
                        idxdiv(occk * oc_bn + ocb, channel_multiplier), ic_bn
                    ),
                    oh * HSTR + rh * dh,
                    ow * WSTR + rw * dw,
                    idxmod(
                        idxdiv(occk * oc_bn + ocb, channel_multiplier), ic_bn
                    ),
                ]
                * W[0, occk, rh, rh, 0, ocb]
            ),
            axis=[rw, rh],
        ),
        name="B",
        tag="depthwise_conv2d_NCHWc",
    )

if open_image ==1:
    B.dtype="climgfloatw32"

# Designate the memory hierarchy
s = te.create_schedule(B.op)
def _schedule_depthwise_conv2d_NCHWc_impl(s, pad_data, kernel, conv, op):
    cfg = autotvm.get_config()
    max_unroll = 16
    vec_size = [1, 2, 4, 8, 16]
    ##### space definition begin #####
    n, c, y, x,_ = s[conv].op.axis
    cfg.define_split("tile_cp", c, num_outputs=2)
    cfg.define_split("tile_hp", y, num_outputs=3)
    cfg.define_split("tile_wp", x, num_outputs=3,filter=lambda y: y.size[-1] >=2)
    Apad,W,B = pad_data, kernel, conv

    # schedule pad and pack
    s[Apad].compute_inline()
    
    #cache
    AL = s.cache_read(Apad, "local", [B])
    WL = s.cache_read(W, "local", [B])
    BL = s.cache_write(B, "local")


    n, cp, hp, wp, Op4 = s[B].op.axis
    block_factor=4
    wpo,wpi=s[B].split(wp,factor=2)
    hpo, hpii = s[B].split(hp, factor=2)
    hpo,hpi=s[B].split(hpo,factor=block_factor)
    wpo,wpi=s[B].split(wpo,factor=block_factor//2)
    cpo, cpi = s[B].split(cp, factor=block_factor * 16)
    
    #cpo, cpi = cfg["tile_cp"].apply(s, B, cp)
    #wpo, wpi, Owp = cfg["tile_wp"].apply(s, B, wp)
    #hpo, hpi, Ohp=cfg["tile_hp"].apply(s, B, hp)

    s[B].bind(cpo, te.thread_axis("blockIdx.z"))
    s[B].bind(cpi, te.thread_axis("threadIdx.z"))
    s[B].bind(hpo, te.thread_axis("blockIdx.y"))
    s[B].bind(hpi, te.thread_axis("threadIdx.y"))
    s[B].bind(wpo, te.thread_axis("blockIdx.x"))
    s[B].bind(wpi, te.thread_axis("threadIdx.x"))

    
    s[B].reorder(wpi, hpo, cpo, hpi, wpo, hpii, cpi)
    s[BL].compute_at(s[B], cpi)
    
    
    k1, k2 = s[BL].op.reduce_axis
    _, kp, hp, wp, p4 = s[BL].op.axis
    s[BL].reorder(hp, wp, p4)
    s[BL].vectorize(p4)
    #s[BL].unroll(wp)
    #s[BL].unroll(hp)
    #s[BL].unroll(k1)
    #s[BL].unroll(k2)

    #schedule UL VL local read
    s[AL].compute_at(s[BL], kp)
    s[WL].compute_at(s[BL], kp)

    #split Ul VL workload
    a, b, hp, wp, _, p4 = s[WL].op.axis
    s[WL].vectorize(p4)  # vectorize memory load
    #s[WL].unroll(wp)
    #s[WL].unroll(hp)
    _, _, hp, wp, p4 = s[AL].op.axis
    s[AL].vectorize(p4)  # vectorize memory load
    #s[AL].unroll(wp)
    #s[AL].unroll(hp)

    s[B].vectorize(Op4)  # vectorize memory load
    #s[B].unroll(Owp)  # vectorize memory load
    #s[B].unroll(Ohp)  # vectorize memory load

    #n, ci, yi, xi = s[BL].op.axis

    #cfg["ann_spatial"].apply(
    #    s,
    #    BL,
    #    [ci, yi, xi],
    #    axis_lens=[cfg["tile_c"].size[2],
    #                cfg["tile_y"].size[2], cfg["tile_x"].size[2]],
    #    max_unroll=max_unroll,
    #    vec_size=vec_size,
    #    cfg=cfg,
    #)
_schedule_depthwise_conv2d_NCHWc_impl(s,Apad,W,B,B.op)


def _origin_schedule(s,Apad,W,B):
    s[Apad].compute_inline()  # compute Apad inline
    #AA = s.cache_read(Apad, "shared", [B])
    #WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(Apad, "local", [B])
    WL = s.cache_read(W, "local", [B])
    BL = s.cache_write(B, "local")

    # tile consts
    tile = 1
    num_thread = 4
    block_factor = tile * num_thread
    step = 4
    vthread = 2

    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis( "threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    thread_xz = te.thread_axis("vthread", name="vx")
    thread_yz = te.thread_axis("vthread", name="vy")

    # Split the workloads
    _,kp, hp, wp,Pp4  = s[B].op.axis
    wpo,wpi=s[B].split(wp,factor=2)
    hpo,hpii=s[B].split(hp,factor=2)


    #s[B].reorder(wpo,hpo,kp,wpi,hpi,p4)
    #whp = s[B].fuse(wpi,hpi)
    #s[B].reorder(wpo,hpo,kp,whp,p4)
    #s[B].vectorize(p4)  # vectorize memory load
    #s[B].unroll(whp)
    #rc, = s[B].op.reduce_axis
    #rco,rci=s[B].split(rc,factor=4)
    #s[B].reorder(wpo,hpo,kp,rco,whp,rci,p4)
    #s[B].vectorize(p4)  # vectorize memory load
    #s[B].unroll(whp)
    #
    #print(tvm.lower(s, [A, W, B], simple_mode=True))
    #exit(0)
    ########
    hpo,hpi=s[B].split(hpo,factor=block_factor)
    wpo,wpi=s[B].split(wpo,factor=block_factor//2)
    kpo,kpi=s[B].split(kp,factor=block_factor*16)

    #target = 'llvm -mcpu=core-avx2'
    #func = tvm.build(s, [A, W, B], target=target, name='mmult')
    #np.random.seed(5)
    #a_np = np.random.uniform(size=(C_P,H_P,W_P,PACK4)).astype("float32")
    #w_np = np.random.uniform(size=(in_channel,out_channel)).astype("float32")
    #ctx = tvm.context(target, 0)
    #a = tvm.nd.array(a_np, ctx,dtype=A.dtype)
    #w = tvm.nd.array(w_np, ctx,dtype=W.dtype)
    #b = tvm.nd.array(np.zeros((K_P, H_P,W_P,PACK4), dtype="float32"), ctx, dtype=B.dtype)
    #func(a, w, b)
    #exit(0)
    #
    # Bind the iteration variables to GPU thread indices
    s[B].bind(hpo, block_y)
    s[B].bind(wpo, block_x)
    s[B].bind(kpo, block_z)

    #tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    #txz, wi = s[B].split(wi, nparts=vthread)  # virtual thread split
    #ty, fi = s[B].split(hpi, nparts=num_thread)
    #tx, wi = s[B].split(wpi, nparts=num_thread)
    #tx, wi = s[B].split(kpi, nparts=num_thread)
    #s[B].reorder(bz, by, bx, tyz, txz, ty, fi, ni)
    #s[B].reorder(bz,by,bx,wi, fi,ty)
    #s[B].reorder(bz,by,bx,ty,wi, fi)

    #s[B].bind(tyz, thread_yz)
    #s[B].bind(txz, thread_xz)
    s[B].bind(hpi, thread_y)
    s[B].bind(wpi, thread_x)
    s[B].bind(kpi, thread_z)
    s[B].reorder(wpi,hpo,kpo,hpi,wpo,hpii,kpi)


    # Schedule BL local write
    s[BL].compute_at(s[B],kpi)
    s[B].reorder(kpi,hpii)
    _,kp, hp, wp,p4  = s[BL].op.axis
    #wpo,wpi=s[BL].split(wp,factor=2)
    #hpo,hpi=s[BL].split(hp,factor=2)

    #wp,p4 = s[BL].split(wp_p4, factor=4)
    s[BL].reorder(hp,wp,p4)
    whp = s[BL].fuse(wp,hp)
    rw,rh = s[BL].op.reduce_axis
    s[BL].reorder(rw,rh,whp,p4)
    s[BL].vectorize(p4)  # vectorize memory load
    #s[BL].unroll(whp)
    #s[BL].unroll(p4)
    #s[BL].unroll(rci)

    # Attach computation to iteration variables
    #s[AA].compute_at(s[BL], rco)
    #s[WW].compute_at(s[BL], rco)
    s[AL].compute_at(s[BL], kp)
    s[WL].compute_at(s[BL], kp)

    _,kp, hp, wp,p4 = s[AL].op.axis

    #wpo, wpi = s[AL].split(wp_p4, factor=4)

    s[AL].vectorize(p4)  # vectorize memory load
    s[AL].unroll(wp)
    s[AL].unroll(hp)
    #s[AL].reorder(wpo,hp)

    # Schedule for W's shared memory load
    _,kp,h,w,_,p4 = s[WL].op.axis
    print(kp,'==========')
    #ty, ci = s[WL].split(ci, nparts=num_thread)
    #_, cpi = s[WL].split(cp, factor=4)
    #tx, fi = s[WW].split(fi, nparts=num_thread)
    #s[WW].reorder(ty, tx, yi, xi, ci, fi)
    #s[WW].bind(ty, thread_y)
    #s[WW].bind(tx, thread_x)
    s[WL].vectorize(p4)  # vectorize memory load
    s[WL].unroll(kp)

    #wpio,wpii = s[B].split(wp4, factor=4)
    s[B].vectorize(Pp4)  # vectorize memory load
    #s[B].unroll(wpio)
    #s[B].unroll(hpii)


#print(tvm.lower(s, [A, W, B], simple_mode=True))
#exit(0)


###############################################################################
# Generate CUDA Kernel
# --------------------
#
# Finally we use TVM to generate and compile the CUDA kernel, and evaluate the
# latency of convolution.
#
print(tvm.lower(s, [A, W, B], simple_mode=True))

target="opencl"
#target="cuda"
func = tvm.build(s, [A, W, B], target)
print("------opencl code------")
#print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
ctx = tvm.context(target, 0)
#ctx = tvm.gpu(0)

H_P=W_P=in_size
ao_np = np.arange(in_channel*in_size*in_size)
#a_np=np.zeros((C_P*PACK4,H_P,W_P))
#a_np[:in_channel,:in_size,:in_size]=ao_np.reshape(in_channel,in_size,in_size)
a_np=ao_np.reshape(in_channel,in_size,in_size)
#a_np=a_np.reshape(C_P*PACK4,H_P*W_P)
a_np=a_np.reshape(C_P*PACK4,H_P*W_P)
wo_np = np.arange(in_channel*out_channel).reshape(out_channel,in_channel)
w_np=np.zeros((out_channel,C_P*PACK4))
w_np[:out_channel,:in_channel]=wo_np
w_np=wo_np

a_tvm=a_np.T.reshape(1,C_P,H_P,W_P,4)
w_tvm = np.arange(in_channel * 9).reshape(1,C_P,3,3,1,4)

a = tvm.nd.array(a_tvm ,ctx,dtype=A.dtype)
w = tvm.nd.array(w_tvm ,ctx,dtype=W.dtype)
b = tvm.nd.array(np.zeros((1,C_P, H_P,W_P,PACK4), dtype="float32"), ctx, dtype=B.dtype)
func(a, w, b)
evaluator = func.time_evaluator(func.entry_name, ctx, number=15)
cost = evaluator(a, w, b).mean
print("\nanswer::correct!,Convolution: %fms, about %f GFLOPS" % (cost*1e3,out_channel*in_channel*in_size*in_size*2/cost/1e9))
