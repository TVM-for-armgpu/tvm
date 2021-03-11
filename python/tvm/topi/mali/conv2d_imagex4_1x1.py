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
    rc, _, _ = s[BL].op.reduce_axis

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

    s[BL].reorder(rco, rcm, rci, whp, p4, kh, kw)
    s[BL].vectorize(p4)  # vectorize memory load
    s[BL].unroll(whp)
    s[BL].unroll(rci)
    s[BL].unroll(rcm)

    s[AL].compute_at(s[BL], rco)
    s[WL].compute_at(s[BL], rco)

    _, kp, hp, wp, p4 = s[AL].op.axis
    wpo, wpi = wp, p4
    s[AL].vectorize(wpi)  # vectorize memory load
    s[AL].unroll(wpo)
    s[AL].unroll(hp)

    # Schedule for W's shared memory load
    kp, _, _, _,_, p4 = s[WL].op.axis
    cpi = p4
    s[WL].vectorize(cpi)  # vectorize memory load
    s[WL].unroll(kp)

    wpio, wpii = wp4, b_p4
    s[B].vectorize(wpii)  # vectorize memory load
    s[B].unroll(wpio)  # vectorize memory load
    s[B].unroll(hpii)  # vectorize memory load
    return s
