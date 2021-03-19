import logging
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.space import get_factors

from ..utils import traverse_inline, get_const_int, get_const_tuple
from .. import nn


def _schedule_conv_NCHWc(s, cfg, data_vec, kernel_vec, conv_out, op):
    Apad = data_vec
    W = kernel_vec
    B = conv_out
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
        n, kp, hp, wp = s[B].op.axis
        p4 = kp
        len_4_flag = True
    else:
        n, kp, hp, wp, p4 = s[B].op.axis
    #==========
    #n, kp, hp, wp, p4 = s[B].op.axis

    n, f, y, x, b_p4 = n, kp, hp, wp, p4

    bf, kpi = cfg["tile_oc"].apply(s, B, f)
    by, vy, hpii = cfg["tile_oh"].apply(s, B, y)
    bx, vx, wp4 = cfg["tile_ow"].apply(s, B, x)

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
    if cfg["cmpat_when_kernel"].val > 0:
        s[BL].pragma(kw, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BL].pragma(kw, "unroll_explicit", cfg["unroll_explicit"].val)
        s[BL].pragma(kh, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BL].pragma(kh, "unroll_explicit", cfg["unroll_explicit"].val)
    else:
        s[BL].pragma(rco, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BL].pragma(rco, "unroll_explicit", cfg["unroll_explicit"].val)

    at_axis = rco
    if cfg["cmpat_when_kernel"].val == 1:
        at_axis = kh
    elif cfg["cmpat_when_kernel"].val == 2:
        at_axis = kw
    s[AL].compute_at(s[BL], at_axis)
    s[WL].compute_at(s[BL], at_axis)

    _, kp, hp, wp, p4 = s[AL].op.axis
    s[AL].vectorize(p4)  # vectorize memory load
    s[AL].unroll(wp)
    s[AL].unroll(hp)

    # Schedule for W's shared memory load
    kp, _, kh, kw,_, p4 = s[WL].op.axis
    cpi = p4
    s[WL].vectorize(cpi)  # vectorize memory load
    s[WL].unroll(kp)
    s[WL].unroll(kh)
    s[WL].unroll(kw)

    wpio, wpii = wp4, b_p4
    s[B].vectorize(wpii)  # vectorize memory load
    s[B].unroll(wpio)  # vectorize memory load
    s[B].unroll(hpii)  # vectorize memory load
    return s
