import logging
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.space import get_factors

from ..utils import traverse_inline, get_const_int, get_const_tuple
from .. import nn

def compute_at_axis_filter(kernel_max):
    if kernel_max > 3:
        return 3
    elif kernel_max > 1:
        return 3
    else:
        return 1
def _schedule_conv_NCHWc_rf(s, cfg, data_vec, kernel_vec, conv_out, op):
    Apad = data_vec
    W = kernel_vec
    B = conv_out
    kernel_height, kernel_width = kernel_vec.shape[2:4]
    kernel_max = max(kernel_height,kernel_width)
    cmpat_when_kernel=compute_at_axis_filter(kernel_max)
    #=========
    if 'conv2d_NCHWc_rf' not in op.tag:
        assert 0, 'failed with schedule, rf is used'
    conv_out_rf = op.output(0)
    #=====scheule rf first, convout->rf is the final result
    RFL = s.cache_write(conv_out_rf, "local")
    if op not in s.outputs:
        s[conv_out_rf].compute_inline()
        conv_out_rf = s.outputs[0].output(0)
    ax_all = s[conv_out_rf].op.axis
    len_4_flag = False
    if len(ax_all) == 4:
        n, f, y, x = s[conv_out_rf].op.axis
        RF_p4 = kp
        len_4_flag = True
    else:
        _,f,y,x,RF_p4 = s[conv_out_rf].op.axis

    bf, kpi = cfg["tile_oc"].apply(s, conv_out_rf, f)
    by, vy = s[conv_out_rf].split(y, factor=cfg["tile_oh"].size[-1])
    bx, vx = s[conv_out_rf].split(x, factor=cfg["tile_ow"].size[-1])

    if len_4_flag:
        kpi,b_p4 = s[conv_out_rf].split(kpi, factor=4)
    #bf, kpi = s[conv_out_rf].split(f, factor=64)
    ##yo, hpii = s[conv_out_rf].split(y, factor=2)
    #by, vy = s[conv_out_rf].split(y, factor=4)
    ##xo, wp4 = s[conv_out_rfB].split(x, factor=2)
    #bx, vx = s[conv_out_rf].split(x, factor=2)

    s[conv_out_rf].bind(bf, te.thread_axis("blockIdx.z"))
    s[conv_out_rf].bind(by, te.thread_axis("blockIdx.y"))
    s[conv_out_rf].bind(bx, te.thread_axis("blockIdx.x"))
    s[conv_out_rf].bind(kpi, te.thread_axis("threadIdx.z"))
    s[conv_out_rf].bind(vy, te.thread_axis("threadIdx.y"))
    s[conv_out_rf].bind(vx, te.thread_axis("threadIdx.x"))
    s[conv_out_rf].reorder(bf, by, bx, vy, vx, kpi)

    s[RFL].compute_at(s[conv_out_rf], kpi)
    s[conv_out_rf].reorder(kpi, vx)

    n_fn,_,_,_,p4 = s[RFL].op.axis
    rf_c, = s[RFL].op.reduce_axis
    s[RFL].vectorize(p4)
    #s[RFL].unroll(rf_c)
    s[conv_out_rf].vectorize(RF_p4)
    #=====scheule rf end

    B = conv_out_rf.op.input_tensors[0]
    _schedule_conv_NCHWc(s, cfg, data_vec, kernel_vec, conv_out, B.op, True)
    return s


def _schedule_conv_NCHWc(s, cfg, data_vec, kernel_vec, conv_out, op, from_rf=False):
    Apad = data_vec
    W = kernel_vec
    B = conv_out
    kernel_height, kernel_width = kernel_vec.shape[2:4]
    kernel_max = max(kernel_height,kernel_width)
    cmpat_when_kernel=compute_at_axis_filter(kernel_max)
    #=========
    B = op.output(0)
    Apad, W = s[B].op.input_tensors
    #==========
    if isinstance(s[Apad].op, tvm.te.ComputeOp) and "pad" in Apad.op.tag:
        s[Apad].compute_inline()
    AL = s.cache_read(Apad, "local", [B])
    WL = s.cache_read(W, "local", [B])
    BL = s.cache_write(B, "local")
    if op not in s.outputs and not from_rf:
        s[B].compute_inline()
        B = s.outputs[0].output(0)
    # Split the workloads
    #==========
    ax_all = s[B].op.axis
    len_4_flag = False
    if len(ax_all) == 4:
        n, kp, hp, wp = s[B].op.axis
        p4 = kp
        len_4_flag = True
    else:
        n, kp, hp, wp, p4 = s[B].op.axis
    #==========
    #n, kp, hp, wp, p4 = s[B].op.axis

    n, f, y, x, b_p4 = n, kp, hp, wp, p4

    f=s[B].fuse(n,f)
    bf, kpi = cfg["tile_oc"].apply(s, B, f)
    by, vy, hpii = cfg["tile_oh"].apply(s, B, y)
    bx, vx, wp4 = cfg["tile_ow"].apply(s, B, x)
    #s[B].reorder(hpii, by, vy)
    #s[B].reorder(wp4, bx, vx)

    if len_4_flag:
        kpi,b_p4 = s[B].split(kpi, factor=4)
    #bf, kpi = s[B].split(f, factor=64)
    #yo, hpii = s[B].split(y, factor=2)
    #by, vy = s[B].split(yo, factor=4)
    #xo, wp4 = s[B].split(x, factor=2)
    #bx, vx = s[B].split(xo, factor=2)


    s[B].bind(bf, te.thread_axis("blockIdx.z"))
    s[B].bind(by, te.thread_axis("blockIdx.y"))
    s[B].bind(bx, te.thread_axis("blockIdx.x"))
    #s[B].bind(hpii, te.thread_axis("vthread", name="vx"))
    #s[B].bind(wp4, te.thread_axis("vthread", name="vy"))
    s[B].bind(kpi, te.thread_axis("threadIdx.z"))
    s[B].bind(vy, te.thread_axis("threadIdx.y"))
    s[B].bind(vx, te.thread_axis("threadIdx.x"))

    s[B].reorder(bf, by, bx, vy, vx, hpii, kpi)
    s[BL].compute_at(s[B], kpi)
    s[B].reorder(kpi, hpii)

    _, kp, hp, wp, p4 = s[BL].op.axis
    whp = s[BL].fuse(wp, hp)
    rc, kh, kw = s[BL].op.reduce_axis

    rco, rcm, rci = cfg["tile_ic"].apply(s, BL, rc)
    #rco, rci = s[BL].split(rc, factor=4)
    #rco, rcm = s[BL].split(rco, factor=1)

    s[BL].reorder(rco, kh, kw, rcm, rci, whp, p4)
    s[BL].vectorize(p4)  # vectorize memory load
    s[BL].unroll(whp)
    s[BL].unroll(rci)
    s[BL].unroll(rcm)

    #s[BL].unroll(kh)
    #s[BL].unroll(kw)
    if kernel_max > 1:
        s[BL].pragma(kw, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BL].pragma(kw, "unroll_explicit", cfg["unroll_explicit"].val)
        s[BL].pragma(kh, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BL].pragma(kh, "unroll_explicit", cfg["unroll_explicit"].val)
        s[BL].pragma(rco, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BL].pragma(rco, "unroll_explicit", cfg["unroll_explicit"].val)


    at_axis = rco
    # theoreticallym  the condition should be cfg["cmpat_when_kernel"].size[-1]-1, but the current would be better
    if cmpat_when_kernel == 2:
        at_axis = kh
    elif cmpat_when_kernel == 3:
        at_axis = kw
    s[AL].compute_at(s[BL], at_axis)
    s[WL].compute_at(s[BL], at_axis)

    _, kp, hp, wp, p4 = s[AL].op.axis
    s[AL].vectorize(p4)  # vectorize memory load
    s[AL].unroll(wp)
    s[AL].unroll(hp)

    # Schedule for W's shared memory load
    kp, _, kh, kw,_, p4 = s[WL].op.axis
    s[WL].vectorize(p4)  # vectorize memory load
    s[WL].unroll(kp)
    s[WL].unroll(kh)
    s[WL].unroll(kw)

    wpio, wpii = wp4, b_p4
    s[B].vectorize(wpii)  # vectorize memory load
    s[B].unroll(wpio)  # vectorize memory load
    s[B].unroll(hpii)  # vectorize memory load
    return s


def _schedule_conv_NHCWc(s, cfg, data_vec, kernel_vec, conv_out, op):
    Apad = data_vec
    W = kernel_vec
    B = conv_out
    kernel_height = kernel_vec.shape[2]
    #=========
    B = op.output(0)
    Apad, W = s[B].op.input_tensors
    #==========
    if isinstance(s[Apad].op, tvm.te.ComputeOp) and "pad" in Apad.op.tag:
        s[Apad].compute_inline()
    AL = s.cache_read(Apad, "local", [B])
    WL = s.cache_read(W, "local", [B])
    BL = s.cache_write(B, "local")
    if op not in s.outputs:
        s[B].compute_inline()
        B = s.outputs[0].output(0)
    # Split the workloads
    #==========
    ax_all = s[B].op.axis
    len_4_flag = False
    if len(ax_all) == 4:
        n, hp, kp, wp = s[B].op.axis
        p4 = kp
        len_4_flag = True
    else:
        n, hp, kp, wp, p4 = s[B].op.axis
    #==========
    #n, kp, hp, wp, p4 = s[B].op.axis

    n, f, y, x, b_p4 = n, kp, hp, wp, p4

    s[B].reorder(n, f, y, x, b_p4)
    f = s[B].fuse(n, f)
    bf, kpi = cfg["tile_oc"].apply(s, B, f)
    by, vy = cfg["tile_oh"].apply(s, B, y)
    bx, vx, hpii, wp4 = cfg["tile_ow"].apply(s, B, x)
    s[B].reorder(hpii, wp4, bx, vx)

    if len_4_flag:
        kpi, b_p4 = s[B].split(kpi, factor=4)
    #bf, kpi = s[B].split(f, factor=64)
    #yo, hpii = s[B].split(y, factor=2)
    #by, vy = s[B].split(yo, factor=4)
    #xo, wp4 = s[B].split(x, factor=2)
    #bx, vx = s[B].split(xo, factor=2)

    s[B].bind(bf, te.thread_axis("blockIdx.x"))
    s[B].bind(by, te.thread_axis("blockIdx.z"))
    s[B].bind(bx, te.thread_axis("blockIdx.y"))
    s[B].bind(hpii, te.thread_axis("vthread", name="vx"))
    s[B].bind(wp4, te.thread_axis("vthread", name="vy"))
    s[B].bind(kpi, te.thread_axis("threadIdx.x"))
    s[B].bind(vy, te.thread_axis("threadIdx.z"))
    s[B].bind(vx, te.thread_axis("threadIdx.y"))

    s[B].reorder(bf, by, bx, vy, vx, hpii, kpi)
    s[BL].compute_at(s[B], kpi)
    s[B].reorder(kpi, hpii)

    _, hp, kp, wp, p4 = s[BL].op.axis
    s[BL].reorder(_, kp, hp, wp, p4)
    whp = s[BL].fuse(wp, hp)
    rc, kh, kw = s[BL].op.reduce_axis

    rco, rcm, rci = cfg["tile_ic"].apply(s, BL, rc)
    #rco, rci = s[BL].split(rc, factor=4)
    #rco, rcm = s[BL].split(rco, factor=1)

    s[BL].reorder(rco, kh, kw, rcm, rci, whp, p4)
    s[BL].vectorize(p4)  # vectorize memory load
    s[BL].unroll(whp)
    s[BL].unroll(rci)
    s[BL].unroll(rcm)

    #s[BL].unroll(kh)
    #s[BL].unroll(kw)
    if kernel_height == 1:
        cfg["unroll_explicit"].val = 1
    else:
        s[BL].pragma(kw, "auto_unroll_max_step",
                     cfg["auto_unroll_max_step"].val)
        s[BL].pragma(kw, "unroll_explicit", cfg["unroll_explicit"].val)
        s[BL].pragma(kh, "auto_unroll_max_step",
                     cfg["auto_unroll_max_step"].val)
        s[BL].pragma(kh, "unroll_explicit", cfg["unroll_explicit"].val)
    s[BL].pragma(rco, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[BL].pragma(rco, "unroll_explicit", cfg["unroll_explicit"].val)

    at_axis = rco
    # theoreticallym  the condition should be cfg["cmpat_when_kernel"].size[-1]-1, but the current would be better
    if cfg["cmpat_when_kernel"].size[-1] == 2:
        at_axis = kh
    elif cfg["cmpat_when_kernel"].size[-1] == 3:
        at_axis = kw
    s[AL].compute_at(s[BL], at_axis)
    s[WL].compute_at(s[BL], at_axis)

    _, hp, kp, wp, p4 = s[AL].op.axis
    s[AL].vectorize(p4)  # vectorize memory load
    s[AL].unroll(wp)
    s[AL].unroll(hp)

    # Schedule for W's shared memory load
    _, kp, kh, kw, _, p4 = s[WL].op.axis
    s[WL].vectorize(p4)  # vectorize memory load
    s[WL].unroll(kp)
    s[WL].unroll(kh)
    s[WL].unroll(kw)

    wpio, wpii = wp4, b_p4
    s[B].vectorize(wpii)  # vectorize memory load
    s[B].unroll(wpio)  # vectorize memory load
    s[B].unroll(hpii)  # vectorize memory load
    return s
