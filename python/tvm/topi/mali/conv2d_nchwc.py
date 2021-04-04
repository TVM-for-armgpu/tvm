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
    conv_out, flops = conv2d_NCHWc_io_op_common(data, kernel, stride, padding,
                                         dilation, layout, out_layout, False,
                                         out_dtype)

    N, oc_chunk, out_height, out_width, ic_bn = get_const_tuple(conv_out.shape)
    if len(kernel.shape) == 6:
        in_channel, _, kernel_height, kernel_width, _, _ = get_const_tuple(kernel.shape)
    else:
        in_channel, _, kernel_height, kernel_width, = get_const_tuple(kernel.shape)
    cfg.is_fallback = False

    # Define autotvm tuning space
    #x.size[-1] == ic_bn must exist, don't change it
    cfg.define_split("tile_ic", in_channel, num_outputs=3,
                     filter=lambda x: x.size[1] <= 12 and x.size[-1] == ic_bn)
    cfg.define_split("tile_oc", oc_chunk, num_outputs=2)
    if kernel_height == 1 and kernel_width == 1:
        ow_lambda = lambda y: y.size[-1] % 2 ==0 and y.size[-1] <= 8
        ow_policy='factors'
    else:
        ow_lambda = lambda y: y.size[-1] <= 8
        ow_policy='verbose'
    cfg.define_split("tile_ow",
                     (1+out_width)//2*2,
                     num_outputs=3,
                     filter=ow_lambda,
                     policy=ow_policy)
    if kernel_height == 1 and kernel_width == 1:
        oh_lambda = lambda y: y.size[-1] % 2 == 0 and y.size[-1] <= 8
        oh_policy='factors'
    else:
        oh_lambda = lambda y: y.size[-1] <= 8
        oh_policy='verbose'
    cfg.define_split(
        "tile_oh",
        (1+out_height)//2*2,
        num_outputs=3,
        filter=oh_lambda,
        policy=oh_policy,
    )
    device_key="Mali"
    #for 3x3 or 5x5 or 7x7 convolution
    kernel_max = max(kernel_height,kernel_width)
    def compute_at_axis_filter():
        #rco, kh, kw--->>>1 2 3
        #    if kernel_max > 3:
        #        return [3]
        #    elif kernel_max > 1:
        #        return [2]
        #    else:
        #        return [1]
        if kernel_max > 3:
            if 'MaliG721' in device_key:
                return [2]
            else:
                return [3]
        elif kernel_max > 1:
            if 'MaliG721' in device_key:
                return [1]
            else:
                return [3]
        else:
            return [1]

    cfg.define_knob("cmpat_when_kernel",compute_at_axis_filter())
    if kernel_max > 1:
        cfg.define_knob("auto_unroll_max_step", [256])
        cfg.define_knob("unroll_explicit", [0, 1])
    # for mali======================

    if 'Mali' in device_key:
        cfg.define_knob("idtype", [0, 1])
        cfg.define_knob("kdtype", [0, 1])
        #cfg.define_knob("odtype", [0, 2])
        typedict = {0: "float32", 1: "climgfloatr32", 2: "climgfloatw32"}
        data.dtype = typedict[cfg["idtype"].val]
        kernel.dtype = typedict[cfg["kdtype"].val]
    #end define autotvm space
    cfg.add_flop(flops)
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
        if 'conv2d_NCHWc_rf' in op.tag:
            conv_out_rf = op.output(0)
            conv_out = conv_out_rf.op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            args = [s, cfg, data_vec, kernel_vec, conv_out, op]
            return conv2d_imagex4_1x1._schedule_conv_NCHWc_rf(*args)
        elif "conv2d_NCHWc" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            #packed_data, packed_kernel = kernel_vec, data_vec
            #if isinstance(packed_kernel.op, tvm.te.ComputeOp) and "packed_kernel" in packed_kernel.name:
            #    # data and kernel are not pre-computed, schedule layout transform here
            #    schedule_injective_from_existing(s, packed_data)
            #    schedule_injective_from_existing(s, packed_kernel)

            args = [s, cfg, data_vec, kernel_vec, conv_out, op]
            (
                _,
                _,
                kh,
                kw,
                _,
                _,
            ) = get_const_tuple(kernel_vec.shape)
            return conv2d_imagex4_1x1._schedule_conv_NCHWc(*args)
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
