# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""conv2d schedule on ARM Mali GPU"""
import logging
import tvm
from tvm import te
from tvm import relay
from tvm import autotvm
from tvm.autotvm.task.space import get_factors
from ..nn.conv2d import unpack_NCHWc_to_nchw

from ..utils import traverse_inline, get_const_int, get_const_tuple
from .. import nn
from ..nn.winograd_util import winograd_transform_matrices
from ..nn.conv2d import conv2d_winograd_nhwc, _conv2d_winograd_nhwc_impl
from . import conv2d_imagex4_1x1
# reuse some compute declarations from ARM CPU
from ..arm_cpu.conv2d_spatial_pack import conv2d_spatial_pack_nchw, conv2d_spatial_pack_nchw_io
from . import op_mad
from . import mali_winograd
from .conv2d import _pack_data
from .conv2d import conv2d_NCHWc_io_op_common
from tvm.topi.cuda.injective import schedule_injective_from_existing
logger = logging.getLogger("topi")


@autotvm.register_topi_compute("conv2d_NCHWc_mali_io.mali")
def conv2d_NCHWc_mali_io(cfg, data, kernel, stride, padding, dilation, layout, out_layout, out_dtype):
    #cfg.define_knob("idtype", [0, 1])
    cfg.define_knob("kdtype", [0, 1])
    #cfg.define_knob("odtype", [0, 2])
    typedict = {0: "float32", 1: "climgfloatr32", 2: "climgfloatw32"}
    #data.dtype = typedict[cfg["idtype"].val]
    kernel.dtype = typedict[cfg["kdtype"].val]

    conv_out, flops = conv2d_NCHWc_io_op_common(data, kernel, stride, padding,
                                                dilation, layout, out_layout,
                                                out_dtype)

    N, oc_chunk, out_height, out_width, ic_bn = get_const_tuple(conv_out.shape)
    if len(kernel.shape) == 6:
        in_channel, _, kernel_height, _, _, _ = get_const_tuple(kernel.shape)
    else:
        in_channel, _, kernel_height, _, _, _ = get_const_tuple(kernel.shape)
    # Define autotvm tuning space
    #x.size[-1] == ic_bn must exist, don't change it
    cfg.define_split("tile_ic", in_channel, num_outputs=3,
                     filter=lambda x: x.size[1] <= 12 and x.size[-1] == ic_bn)
    #cfg.define_split("tile_oc", oc_chunk, num_outputs=2)
    cfg.define_split("tile_ow",
                     out_width,
                     num_outputs=2,
                     filter=lambda y: y.size[-1]>=2 and y.size[-1] <= 16,
                     policy="verbose")
    cfg.define_split(
        "tile_oh",
        out_height,
        num_outputs=2,
        filter=lambda y: y.size[-1] >= 2 and y.size[-1] <= 16,
        policy="verbose",
    )
    cfg.is_fallback = False
    #for 3x3 or 5x5 or 7x7 convolution
    def compute_at_axis_filter(y):
        if kernel_height > 1:
            # TODO make y.size[-1] could be [2, 3],
            # but now feature for xgboost is not match(diffent feature len),dand training failed
            return y.size[-1] in [2]
        elif kernel_height > 1:
            return y.size[-1] in [2]
        else:
            return y.size[-1] in [1]

    # actual you should set len_size as 6, because 2*3==6,when set as 2,  kernel_height==2 or 3, y.size[-1]==2
    cfg.define_split("cmpat_when_kernel",
                     6,
                     num_outputs=2,
                     filter=compute_at_axis_filter)

    cfg.define_knob("auto_unroll_max_step", [0, 128, 256, 512])
    # for mali======================
    cfg.define_knob("unroll_explicit", [0, 1])

    #end define autotvm space
    cfg.add_flop(flops)
    conv_out.dtype = out_dtype
    #for mali=============
    #conv_out.dtype = typedict[cfg["odtype"].val]
    return conv_out


@autotvm.register_topi_schedule("conv2d_NCHWc_mali_io.mali")
def schedule_conv2d_NCHWc_mali_io(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    def _callback(tensor):
        #op = tensor.op
        op=tensor
        if "conv2d_NCHWc" in op.tag:
            conv_out = op.output(0)
            _,oc_chunk,OH,OW,_ = get_const_tuple(conv_out.shape)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
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
                n, kp, hp, wp = s[B].op.axis
                p4 = kp
                len_4_flag = True
            else:
                n, kp, hp, wp, p4 = s[B].op.axis
            #==========
            #n, kp, hp, wp, p4 = s[B].op.axis

            n, f, y, x, b_p4 = n, kp, hp, wp, p4

            #bf, kpi = cfg["tile_oc"].apply(s, B, f)
            by_vy, hpii = cfg["tile_oh"].apply(s, B, y)
            bx_vx, wp4 = cfg["tile_ow"].apply(s, B, x)

            if len_4_flag:
                kpi,b_p4 = s[B].split(kpi, factor=4)
            #bf, kpi = s[B].split(f, factor=64)
            #yo, hpii = s[B].split(y, factor=2)
            #by, vy = s[B].split(yo, factor=4)
            #xo, wp4 = s[B].split(x, factor=2)
            #bx, vx = s[B].split(xo, factor=2)

            cfg.define_split("tile_bindthread",
                             OH*OW//(cfg["tile_oh"].size[-1] * cfg["tile_ow"].size[-1]) *
                             oc_chunk,
                             num_outputs=6)
            s[B].reorder(n, f, by_vy, bx_vx, hpii, wp4)
            fuse_bind = s[B].fuse(n, f, by_vy, bx_vx)
            bf, by, bx, kpi, vy, vx = cfg["tile_bindthread"].apply(s, B, fuse_bind)

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
            if kernel_height == 1:
                cfg["unroll_explicit"].val = 1
            else:
                s[BL].pragma(kw, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
                s[BL].pragma(kw, "unroll_explicit", cfg["unroll_explicit"].val)
                s[BL].pragma(kh, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
                s[BL].pragma(kh, "unroll_explicit", cfg["unroll_explicit"].val)
            s[BL].pragma(rco, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
            s[BL].pragma(rco, "unroll_explicit", cfg["unroll_explicit"].val)


            at_axis = rco
            # theoreticallym  the condition should be cfg["cmpat_when_kernel"].size[-1]-1, but the current would be better
            if cfg["cmpat_when_kernel"].size[-1] == 2:
                at_axis = kh
            elif cfg["cmpat_when_kernel"].size[-1]  == 3:
                at_axis = kw
            s[AL].compute_at(s[BL], at_axis)
            s[WL].compute_at(s[BL], at_axis)

            _, kp, hp, wp, p4 = s[AL].op.axis
            s[AL].vectorize(p4)  # vectorize memory load
            #s[AL].unroll(wp)
            #s[AL].unroll(hp)
            s[AL].bind(wp, te.thread_axis("blockIdx.x"))
            s[AL].bind(hp, te.thread_axis("threadIdx.x"))

            # Schedule for W's shared memory load
            kp, _, kh, kw,_, p4 = s[WL].op.axis
            s[WL].vectorize(p4)  # vectorize memory load
            s[WL].unroll(kp)
            s[WL].unroll(kh)
            s[WL].unroll(kw)

            wpio, wpii = wp4, b_p4
            s[B].vectorize(wpii)  # vectorize memory load
            #s[B].unroll(wpio)  # vectorize memory load
            #s[B].unroll(hpii)  # vectorize memory load
            s[B].bind(wpio, te.thread_axis("blockIdx.z"))
            s[B].bind(hpii, te.thread_axis("threadIdx.z"))
            return s

    ret_s = None
    scheduled_ops = []
    def traverse(out_tensor) -> bool:
        # _callback end

        op = out_tensor.op
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tvm.topi.tag.is_broadcast(op.tag) or tvm.topi.tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            #for tensor in op.input_tensors:
            #    if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
            #        traverse(tensor)
        if "conv2d_NCHWc" in op.tag:
            ret_s = _callback(out_tensor.op)
            return True

        for tensor in op.input_tensors:
            if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                scheduled_ops.append(op)
                if True == traverse(tensor):
                    return True
        return False
    #traverse_inline(s, outs[0].op, _callback)

    assert traverse(
        outs[0]) != False, "sch is None, please check you compute body"
    return s


def conv2d_nchw_winograd_nchwc_conv3x3_mali_io(data,
                                               kernel,
                                               strides,
                                               padding,
                                               dilation,
                                               out_dtype="float32"):
    """Compute conv2d internally using conv2d_nchwc layout for any dtype"""
    packed_out = conv2d_nchw_winograd_NCHWc_mali_io(data, kernel, strides,
                              padding, dilation, "NCHW", "NCHW4c", out_dtype)

    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


@autotvm.register_topi_compute("conv2d_nchw_winograd_NCHWc_mali_io.mali")
def conv2d_nchw_winograd_NCHWc_mali_io(cfg, data, kernel, strides, padding,
                                       dilation, layout, out_layout,
                                       out_dtype):
    #assert (
    #    out_dtype == 'climgfloatw32' and data.dtype == 'climgfloatr32' and kernel.dtype == 'climgfloatr32'
    #), "only support opencl image type yet"

    #tile_size = _pick_tile_size(data, kernel)
    tile_size = 2
    if len(data.shape) == 5:
        N, ic_chunk, IH, IW, ic_bn = get_const_tuple(data.shape)
        CI = ic_chunk * ic_bn
        ic_chunk, oc_chunk, KH, KW, _, oc_bn = get_const_tuple(kernel.shape)
        CO = oc_chunk * oc_bn
    else:
        N, CI, IH, IW = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        ic_chunk = CI
        oc_chunk = CO
        oc_bn = 1
        ic_bn = 1
    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    pre_computed = False
    if len(data.shape) == 4:
        ic_bn = 4
        oc_bn = 4
        oc_chunk = oc_chunk // oc_bn
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (N, ic_chunk, IH, IW, ic_bn)
            data = tvm.te.placeholder(dshape, data.dtype, name="placeholder")
            kshape = (
                CI,
                CO // 4,
                KH,
                KW,
                1,
                oc_bn,
            )
            kernel = tvm.te.placeholder(kshape,
                                        kernel.dtype,
                                        name="placeholder")
        else:
            data, kernel = _pack_data(cfg, data, kernel)
    else:
        oc_chunk = CO // oc_bn
        if KH == 4 or KH == 6:
            pre_computed = True
    wino = tile_size
    winop2 = tile_size + 2
    wino_size = winop2
    H_DIV_WINO = (IH + wino - 1) // wino
    W_DIV_WINO = (IW + wino - 1) // wino
    wino2H = H_DIV_WINO
    wino2W = W_DIV_WINO
    winop2_tile_size = winop2 * winop2
    NTILE = H_DIV_WINO * W_DIV_WINO
    # 'same' for convolution
    out_shape = (N, oc_chunk, IH, IW, oc_bn)

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    assert dilation_h == 1 and dilation_w == 1

    #if len(kernel.shape) == 4:
    #    if dilation_h != 1 or dilation_w != 1:
    #        kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))
    #    pre_computed = False
    #    CO, _, KH, KW = get_const_tuple(kernel.shape)
    #else:
    #    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
    #    pre_computed = True
    #    H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
    #    CO *= VC
    #    KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    #====================useless temperary===========begin
    HSTR, WSTR = strides if isinstance(
        strides, (tuple, list)) else (strides, strides)
    pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))

    assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype.replace("climg","").replace("w",""))
    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW

    ##### space definition begin #####
    cfg.define_split("inv_cp", oc_chunk, num_outputs=2, max_factor=8)
    cfg.define_split("inv_wp", NTILE, num_outputs=3)
    #, filter=lambda x: x.size[-1] == 4)
    cfg.define_split("inv_hp", winop2_tile_size, num_outputs=3,
                     filter=lambda x: x.size[-1] == 4)

    cfg.define_split("kernel_cp", CI, num_outputs=2, max_factor=32)
    cfg.define_split("kernel_kp", oc_chunk, num_outputs=2, max_factor=32)

    cfg.define_split("data_cp", oc_chunk, num_outputs=2, max_factor=32)
    cfg.define_split("data_wp", oc_chunk, num_outputs=3,
                     filter=lambda x: x.size[-1] % 4 == 0)
    cfg.define_split("data_hp", oc_chunk, num_outputs=3,
                     filter=lambda x: x.size[-1] % 4 == 0)

    cfg.define_split("bgemm_kp", oc_chunk, num_outputs=2, max_factor=32)
    cfg.define_split("bgemm_wp", oc_chunk, num_outputs=3,
                     filter=lambda x: x.size[-1] % 4 == 0)
    cfg.define_split("bgemm_hp", oc_chunk, num_outputs=2)
    ##### space definition end #####

    if pre_computed:
        U = kernel
    else:
        U = mali_winograd.kernel_transform(kernel, wino_size, G=G, out_dtype=out_dtype)
    V = mali_winograd.data_transform(data, wino_size, B=B, out_dtype=out_dtype)
    M = mali_winograd.batch_gemm(U, V, out_dtype=out_dtype)
    output = mali_winograd.inverse_transform(
        out_shape, A=A, M=M, out_dtype=out_dtype)


    output.dtype = out_dtype
    #====================useless temperary===========begin
    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CO * H * W * KH * KW * CI)
    return output


@autotvm.register_topi_schedule("conv2d_nchw_winograd_NCHWc_mali_io.mali")
def schedule_conv2d_nchw_winograd_NCHWc_mali_io(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_conv2d_output" in op.tag:
            mali_winograd._schedule_winograd_nchwc_io(cfg, s, op)
    scheduled_ops = []
    def traverse(out_tensor) -> bool:
        # _callback end

        op = out_tensor.op
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tvm.topi.tag.is_broadcast(op.tag) or tvm.topi.tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
        if "winograd_conv2d_output" in op.tag:
            ret_s = _callback(out_tensor.op)
            return True

        for tensor in op.input_tensors:
            if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                scheduled_ops.append(op)
                if True == traverse(tensor):
                    return True
        return False
    #traverse_inline(s, outs[0].op, _callback)

    assert traverse(
        outs[0]) != False, "sch is None, please check you compute body"
    return s
