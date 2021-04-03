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

from tvm.topi.cuda.injective import schedule_injective_from_existing
logger = logging.getLogger("topi")


@autotvm.register_topi_compute("conv2d_nchw_spatial_pack_io.mali")
def conv2d_nchw_spatial_pack_io(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """TOPI compute callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        pre-packed 5-D with shape [num_filter_chunk, in_channel, filter_height,
        filter_width, num_filter_block]

    strides : list of two ints
        [stride_height, stride_width]

    padding : list of two ints
        [pad_height, pad_width]

    dilation : list of two ints
        [dilation_height, dilation_width]

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return conv2d_spatial_pack_nchw_io(
        cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=3
    )
@autotvm.register_topi_compute("conv2d_nchw_spatial_pack.mali")
def conv2d_nchw_spatial_pack(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """TOPI compute callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    kernel : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        pre-packed 5-D with shape [num_filter_chunk, in_channel, filter_height,
        filter_width, num_filter_block]

    strides : list of two ints
        [stride_height, stride_width]

    padding : list of two ints
        [pad_height, pad_width]

    dilation : list of two ints
        [dilation_height, dilation_width]

    out_dtype: str
        The output type. This is used for mixed precision.

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    return conv2d_spatial_pack_nchw(
        cfg, data, kernel, strides, padding, dilation, out_dtype, num_tile=3
    )


@autotvm.register_topi_schedule("conv2d_nchw_spatial_pack.mali")
def schedule_conv2d_nchw_spatial_pack(cfg, outs):
    """TOPI schedule callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d
    """
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if "spatial_conv2d_output" in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == "kernel_vec":
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            _schedule_spatial_pack(cfg, s, output, conv, data_vec, kernel_vec)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_schedule("conv2d_nchw_spatial_pack_io.mali")
def schedule_conv2d_nchw_spatial_pack_io(cfg, outs):
    """TOPI schedule callback for conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d
    """
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        # schedule conv2d
        if "spatial_conv2d_output" in op.tag:
            output = op.output(0)
            conv = op.input_tensors[0]

            data_vec = conv.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            s[data_pad].compute_inline()

            kernel_vec = conv.op.input_tensors[1]
            if kernel_vec.op.name == "kernel_vec":
                kernel = kernel_vec.op.input_tensors[0]
            else:
                kernel = kernel_vec
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            _schedule_spatial_pack(cfg, s, output, conv, data_vec, kernel_vec)

    traverse_inline(s, outs[0].op, _callback)
    return s
def _schedule_spatial_pack(cfg, s, output, conv, data_vec, kernel_vec):
    """schedule the spatial packing for conv2d"""
    data = s[data_vec].op.input_tensors[0]

    max_unroll = 16
    vec_size = [1, 2, 4, 8, 16]
    # get tunable parameters (they are defined in compute)
    BC, TC, VC = cfg["tile_co"].size
    BH, TH, VH = cfg["tile_oh"].size
    BW, TW, VW = cfg["tile_ow"].size

    # schedule padding
    if isinstance(data.op, tvm.te.ComputeOp) and "pad" in data.op.tag:
        data_pad = data
        s[data_pad].compute_inline()

    # schedule data packing
    if isinstance(data_vec.op, tvm.te.ComputeOp) and data_vec.op.name == "data_vec_undilated":
        _, h, w, ci, _, _, vh, vw = s[data_vec].op.axis
    else:
        _, h, w, ci, vh, vw = s[data_vec].op.axis
    tile_and_bind3d(s, data_vec, h, w, ci, 1)
    if vh.dom.extent.value < max_unroll:
        s[data_vec].unroll(vh)
    if vw.dom.extent.value < max_unroll:
        s[data_vec].unroll(vw)

    if isinstance(kernel_vec.op, tvm.te.ComputeOp) and kernel_vec.name == "kernel_vec":
        if not autotvm.GLOBAL_SCOPE.in_tuning:
            max_threads = tvm.target.Target.current(allow_none=False).max_num_threads
            co, ci, kh, kw, vc = s[kernel_vec].op.axis
            fused = s[kernel_vec].fuse(co, ci, kh, kw, vc)
            fused, vec = s[kernel_vec].split(fused, VC)
            bb, tt = s[kernel_vec].split(fused, max_threads)
            s[kernel_vec].bind(bb, te.thread_axis("blockIdx.x"))
            s[kernel_vec].bind(tt, te.thread_axis("threadIdx.x"))
            if VC in vec_size:
                s[kernel_vec].vectorize(vec)

    # schedule convolution
    n, c, h, w, vh, vw, vc = s[conv].op.axis
    kc, kh, kw = s[conv].op.reduce_axis

    cfg["reorder_0"].apply(s, conv, [n, c, h, w, kc, kh, kw, vh, vw, vc])
    tile_and_bind3d(s, conv, c, h, w, TC, TH, TW)

    cfg["ann_reduce"].apply(
        s,
        conv,
        [kh, kw],
        axis_lens=[get_const_int(kernel_vec.shape[2]), get_const_int(kernel_vec.shape[3])],
        max_unroll=max_unroll,
    )

    cfg["ann_spatial"].apply(
        s,
        conv,
        [vh, vw, vc],
        axis_lens=[VH, VW, VC],
        max_unroll=max_unroll,
        vec_size=vec_size,
        cfg=cfg,
    )

    # schedule output
    if output.op not in s.outputs:  # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    _, co, oh, ow = s[output].op.axis
    tile_and_bind3d(s, output, co, oh, ow, TC, TH, TW)

    return s


##### WINOGRAD TEMPLATE #####
def _pick_tile_size(data, kernel, layout="NCHW"):
    if layout == "NCHW":
        N, CI, H, W = get_const_tuple(data.shape)
    else:
        assert layout == "NHWC"
        N, H, W, CI = get_const_tuple(data.shape)

    if H % 4 == 0:
        return 4
    else:
        return 2


def _pack_data(data, kernel):
    n, _, ih, iw = get_const_tuple(data.shape)
    ic, oc, kh, kw = get_const_tuple(kernel.shape)
    ic_bn, oc_bn = 4, 4

    ic_chunk = (ic+ic_bn-1) // ic_bn
    oc_chunk = (oc+oc_bn-1) // oc_bn

    # Handle dynamic shape to pass tuning dispatch.
    if isinstance(n, tvm.tir.Any):
        n = tvm.te.size_var("n")
    if isinstance(ih, tvm.tir.Any):
        ih = tvm.te.size_var("ih")
    if isinstance(iw, tvm.tir.Any):
        iw = tvm.te.size_var("iw")
    if isinstance(ic, tvm.tir.Any):
        raise RuntimeError("Dynamic input channel is not supported for conv2d.")

    data = te.compute(
        (n, ic_chunk, ih, iw, ic_bn),
        lambda bs, c, h, w, vc: data[bs, c * ic_bn + vc, h, w],
        name="data_vec_nchwc_packed_kernel",
    )
    ic_chunk = ic_chunk*ic_bn
    ic_bn = 1
    kernel = te.compute(
        (ic_chunk, oc_chunk, kh, kw, ic_bn, oc_bn),
        lambda icc, occ, k_h, k_w, icb, ocb: kernel[icc *
                                                    ic_bn + icb, occ * oc_bn + ocb, k_h, k_w],
        name="kernel_vec_nchwc_packed_kernel",
    )

    return data, kernel


@autotvm.register_topi_compute("conv2d_NCHWc12.mali")
def conv2d_NCHWc12(cfg,data, kernel, stride, padding, dilation, layout, out_layout, out_dtype="float32"):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """

    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    target = tvm.target.Target.current(allow_none=False)
    oc_chunk, ic_chunk_group, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)
    num_filter = oc_chunk * oc_bn
    groups = ic_chunk // ic_chunk_group

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = 0,0,0,0
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)

    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD:
        data_pad = pad(data, pad_before, pad_after, name="data_pad")
    else:
        data_pad = data

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    return te.compute(
        oshape,
        lambda n, oc_chunk, oh, ow, oc_block: te.sum(
            data_pad[
                n,
                idxdiv(ic, ic_bn),
                oh * HSTR + kh * dilation_h,
                ow * WSTR + kw * dilation_w,
                idxmod(ic, ic_bn),
            ].astype(out_dtype)
            * kernel[oc_chunk, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block],
            axis=[ic, kh, kw],
        ),
        name="conv2d_NCHWc",
        tag="conv2d_NCHWc",
    )


def conv2d_nchw_conv1x1(data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d internally using conv2d_nchwc layout for any dtype"""
    packed_out = conv2d_NCHWc(data, kernel, strides,
                              padding, dilation, "NCHW", "NCHW4c", out_dtype)
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)

def conv2d_nchw_conv1x1_io(data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d internally using conv2d_nchwc layout for any dtype"""
    packed_out = conv2d_NCHWc_io(data, kernel, strides,
                              padding, dilation, "NCHW", "NCHW4c", out_dtype)

    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


@autotvm.register_topi_compute("conv2d_NCHWc.mali")
def conv2d_NCHWc(cfg, data, kernel, stride, padding, dilation, layout, out_layout, out_dtype):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    """Compute conv2d with NCHWc layout."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, _, kernel_height, kernel_width, _, _ = get_const_tuple(
            kernel.shape)
        #assert kernel_height==1 and kernel_width==1, "only for conv1x1"
        if ic_chunk == 0:
            ic_chunk = 1
        if oc_chunk == 0:
            ic_chunk = 1
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk
        oc_chunk=oc_chunk//ic_bn
        oc_bn=ic_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(
            kernel.shape)
        oc_bn = 4
        ic_bn = 4
        if in_channel == 0:
            in_channel = 1
        if num_filter == 0:
            num_filter = 1
        ic_chunk = (in_channel + 3) // ic_bn
        oc_chunk = (num_filter + 3) // oc_bn
        in_channel = ic_chunk * ic_bn

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, ic_chunk, ih, iw, ic_bn)
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (
                num_filter,
                ic_chunk,
                kernel_height,
                kernel_width,
                1,
                ic_bn,
            )
            kernel = tvm.te.placeholder(kshape,
                                        kernel.dtype,
                                        name="kernel_vec")
        else:
            data, kernel = _pack_data(data, kernel)

    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride,
                                      (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (dilation if isinstance(dilation,
                                                     (tuple, list)) else
                              (dilation, dilation))
    assert (  #kernel_height == 1 and kernel_width == 1 and
        (dilation_h == 1 or dilation_h == 0)
        and (dilation_w == 1 or dilation_w == 0)), " only conv 1x1 support"

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = padding if isinstance(
        padding, (tuple, list)) else (padding, padding, padding, padding)
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)
    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD and kernel_height > 1:
        data_pad = nn.pad(data, pad_before, pad_after, name="data_pad_3x3")
    else:
        data_pad = data
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    conv_out = te.compute(
        oshape,
        lambda n, oc_chunk, oh, ow, oc_block: op_mad.mad(
            op_mad.mymul(
                data_pad[n,
                         idxdiv(ic, ic_bn), oh * HSTR + kh * dilation_h, ow *
                         WSTR + kw * dilation_w,
                         idxmod(ic, ic_bn), ].astype(data.dtype), kernel[
                            oc_chunk*4+oc_block,idxdiv(ic, ic_bn), kh, kw, 0, idxmod(ic, ic_bn)]),
            axis=[ic, kh, kw],
        ),
        name="conv2d_NCHWc",
        tag="conv2d_NCHWc",
    )
    conv_out.dtype = out_dtype
    flops = n * oc_chunk * out_height * out_width * oc_bn * kernel_height * kernel_width * ic_chunk * ic_bn * 2
    # Define autotvm tuning space
    #x.size[-1] == ic_bn must exist, don't change it
    cfg.define_split("tile_ic",
                     in_channel,
                     num_outputs=3,
                     filter=lambda x: x.size[1] <= 12 and x.size[-1] == ic_bn)
    cfg.define_split("tile_oc", oc_chunk, num_outputs=2)
    cfg.define_split("tile_ow",
                     out_width,
                     num_outputs=3,
                     filter=lambda y: y.size[-1] <= 8,
                     policy="verbose")
    cfg.define_split(
        "tile_oh",
        out_height,
        num_outputs=3,
        filter=lambda y: y.size[-1] <= 8,
        policy="verbose",
    )

    #for 3x3 or 5x5 or 7x7 convolution
    def compute_at_axis_filter(y):
        if kernel_height > 1:
            # TODO make y.size[-1] could be [2, 3],
            # but now feature for xgboost is not match(diffent feature len),dand training failed
            return y.size[-1] in [3]
        elif kernel_height > 1:
            return y.size[-1] in [3]
        else:
            return y.size[-1] in [1]

    cfg.define_split("cmpat_when_kernel",
                     6,
                     num_outputs=2,
                     filter=compute_at_axis_filter)

    cfg.define_knob("auto_unroll_max_step", [0, 128, 256, 512])
    # for mali======================
    cfg.define_knob("unroll_explicit", [0, 1])
    cfg.define_knob("idtype", [0, 1])
    cfg.define_knob("kdtype", [0, 1])
    cfg.define_knob("odtype", [0, 2])
    typedict = {0: "float32", 1: "climgfloatr32", 2: "climgfloatw32"}
    data.dtype = typedict[cfg["idtype"].val]
    kernel.dtype = typedict[cfg["kdtype"].val]
    #end define autotvm space
    cfg.add_flop(flops)
    #for mali=============
    conv_out.dtype = typedict[cfg["odtype"].val]
    return conv_out


@autotvm.register_topi_compute("conv2d_NHCWc.mali")
def conv2d_NHCWc_oi4o(cfg, data, kernel, stride, padding, dilation, layout, out_layout, out_dtype):
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ih, ic_chunk, iw, ic_bn = get_const_tuple(data.shape)
        oc_chunk, in_channel, kernel_height, kernel_width, _, oc_bn = get_const_tuple(
            kernel.shape)
        if ic_chunk == 0:
            ic_chunk = 1
        if oc_chunk == 0:
            ic_chunk = 1
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        n, ih, in_channel, iw = get_const_tuple(data.shape)
        num_filter, _, kernel_height, kernel_width = get_const_tuple(
            kernel.shape)
        oc_bn = 4
        ic_bn = 4
        if in_channel == 0:
            in_channel = 1
        if num_filter == 0:
            num_filter = 1
        ic_chunk = (in_channel + 3) // ic_bn
        oc_chunk = (num_filter + 3) // oc_bn
        in_channel = ic_chunk * ic_bn
    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, ih, ic_chunk, iw, ic_bn)
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (
                oc_chunk,
                in_channel,
                kernel_height,
                kernel_width,
                1,
                oc_bn,
            )
            kernel = tvm.te.placeholder(
                kshape, kernel.dtype, name="kernel_vec")
        else:
            #TODO err kernel laytout
            data, kernel = _pack_data(data, kernel)


    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )
    assert (#kernel_height == 1 and kernel_width == 1 and
            (dilation_h == 1 or dilation_h == 0) and
            (dilation_w == 1 or dilation_w == 0)), " only conv 1x1 support"

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = padding if isinstance(
        padding, (tuple, list)) else (padding, padding, padding, padding)
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)
    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD and kernel_height > 1:
        data_pad = nn.pad(data, pad_before, pad_after, name="data_pad_3x3")
    else:
        data_pad = data
    oshape = (n, out_height, oc_chunk, out_width, oc_bn)
    # Define autotvm tuning space
    #x.size[-1] == ic_bn must exist, don't change it
    cfg.define_split("tile_ic",
                     in_channel,
                     num_outputs=3,
                     filter=lambda x: x.size[1] <= 12 and x.size[-1] == ic_bn)
    cfg.define_split("tile_oc", oc_chunk, num_outputs=2)
    cfg.define_split("tile_ow",
                     out_width,
                     num_outputs=4,
                     filter=lambda y: y.size[-1] <= 4 and y.size[-2] <= 4,# for vthread
                     policy="verbose")
    cfg.define_split(
        "tile_oh",
        out_height,
        num_outputs=2,
        policy="verbose",
    )

    #for 3x3 or 5x5 or 7x7 convolution
    def compute_at_axis_filter(y):
        if kernel_height > 1:
            # TODO make y.size[-1] could be [2, 3],
            # but now feature for xgboost is not match(diffent feature len),dand training failed
            return y.size[-1] in [3]
        elif kernel_height > 1:
            return y.size[-1] in [3]
        else:
            return y.size[-1] in [1]

    cfg.define_split("cmpat_when_kernel",
                     6,
                     num_outputs=2,
                     filter=compute_at_axis_filter)

    cfg.define_knob("auto_unroll_max_step", [0, 128, 256, 512])
    # for mali======================
    cfg.define_knob("unroll_explicit", [0, 1])
    cfg.define_knob("idtype", [0, 1])
    cfg.define_knob("kdtype", [0, 1])
    #cfg.define_knob("odtype", [0, 2])
    typedict = {0: "float32", 1: "climgfloatr32", 2: "climgfloatw32"}
    data.dtype = typedict[cfg["idtype"].val]
    kernel.dtype = typedict[cfg["kdtype"].val]

    ic = te.reduce_axis((0, in_channel), name="ic")
    kh = te.reduce_axis((0, kernel_height), name="kh")
    kw = te.reduce_axis((0, kernel_width), name="kw")

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    def get_factors_for_rf():
        for factor in range(4, 8):
            if (ic_chunk//factor)*factor == ic_chunk:
                return factor
        return 1
    rf_n= get_factors_for_rf()
    ic_rf_thread = (ic_chunk//rf_n)*ic_bn
    use_rf=False
    if ic_rf_thread != in_channel and use_rf:
        assert n==1,'only batch==1 is support'
        rf_oshape = (n, out_height, oc_chunk, out_width*rf_n, oc_bn)

        irf = te.reduce_axis((0, rf_n), name="irf")
        ic = te.reduce_axis((0, ic_rf_thread), name="ic")
        kh = te.reduce_axis((0, kernel_height), name="kh")
        kw = te.reduce_axis((0, kernel_width), name="kw")

        idxdiv = tvm.tir.indexdiv
        idxmod = tvm.tir.indexmod
        conv_out_s1 = te.compute(
            rf_oshape,
            lambda n, oc_chunk, oh, ow, oc_block: op_mad.mad(
                op_mad.mymul(
                    data_pad[n, oh * HSTR + kh * dilation_h,
                            idxdiv((ow%rf_n)*ic_rf_thread +ic, ic_bn), ow * WSTR + kw * dilation_w,
                            idxmod(ic, ic_bn), ], kernel[oc_chunk, (ow%rf_n)*ic_rf_thread +ic, kh, kw, 0,
                                                        oc_block]),
                axis=[ic, kh, kw],
            ),
            name="conv2d_NHCWc_nhcw4_s1",
            tag="conv2d_NHCWc_nhcw4",
        )
        #conv_out=conv_out_s1
        conv_out = te.compute(
            oshape,
            lambda n_,  oh, oc_chunk_, ow, oc_block: te.sum(
                    conv_out_s1[
                        n_,
                        oh,
                        oc_chunk_,
                        ow//rf_n+irf,
                        oc_block,
                    ].astype(data.dtype),
                    axis=[irf],
                ),

            name="conv2d_NHCWc_nhcw4",
            tag="conv2d_NHCWc_mace_rf",
        )
    else:
        conv_out = te.compute(
            oshape,
            lambda n, oh, oc_chunk, ow, oc_block: op_mad.mad(
                op_mad.mymul(
                    data_pad[n, oh * HSTR + kh * dilation_h,
                            idxdiv(ic, ic_bn), ow * WSTR + kw * dilation_w,
                            idxmod(ic, ic_bn), ], kernel[oc_chunk, ic, kh, kw, 0,
                                                        oc_block]),
                axis=[ic, kh, kw],
            ),
            name="conv2d_NHCWc_nhcw4",
            tag="conv2d_NHCWc_mace",
        )
    flops = n * oc_chunk * out_height * out_width * oc_bn * kernel_height * kernel_width * ic_chunk * ic_bn * 2
    cfg.add_flop(flops)
    return conv_out


@autotvm.register_topi_schedule("conv2d_NHCWc.mali")
def schedule_conv2d_NHCWc_oi4o(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    def _callback(tensor):
        #op = tensor.op
        op=tensor
        if "conv2d_NHCWc" in op.tag:
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            args = [s, cfg, data_vec, kernel_vec, conv_out, op]
            return conv2d_imagex4_1x1._schedule_conv_NHCWc(*args)

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
        if "conv2d_NHCWc" in op.tag:
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

def conv2d_NCHWc_io_op_common(data, kernel, stride, padding, dilation, layout, out_layout, use_rf, out_dtype):
    """Conv2D operator for nChw[x]c layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.te.Tensor
        6-D with shape
        [num_filter_chunk, in_channel_chunk, filter_height, filter_width,
        in_channel_block, num_filter_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    out_dtype : str
        output data type

    Returns
    -------
    output : tvm.te.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    """Compute conv2d with NCHWc layout."""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    if len(data.shape) == 5:
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        ic_chunk_group, oc_chunk, kernel_height, kernel_width, _, oc_bn = get_const_tuple(
            kernel.shape
        )
        #assert kernel_height==1 and kernel_width==1, "only for conv1x1"
        if ic_chunk == 0:
            ic_chunk=1
        if oc_chunk == 0:
            ic_chunk=1
        in_channel = ic_chunk * ic_bn
        num_filter = oc_chunk * oc_bn
    else:
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        _, num_filter, kernel_height, kernel_width = get_const_tuple(
            kernel.shape)
        oc_bn = 4
        ic_bn = 4
        if in_channel == 0:
            in_channel=1
        if num_filter == 0:
            num_filter=1
        ic_chunk = (in_channel+3)//ic_bn
        oc_chunk = (num_filter+3)//oc_bn
        in_channel = ic_chunk * ic_bn

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (n, ic_chunk, ih, iw, ic_bn)
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (
                in_channel,
                oc_chunk,
                kernel_height,
                kernel_width,
                1,
                oc_bn,
            )
            kernel = tvm.te.placeholder(
                kshape, kernel.dtype, name="kernel_vec")
        else:
            data, kernel = _pack_data(data, kernel)


    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HSTR, WSTR = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    dilation_h, dilation_w = (
        dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    )
    assert (#kernel_height == 1 and kernel_width == 1 and
            (dilation_h == 1 or dilation_h == 0) and
            (dilation_w == 1 or dilation_w == 0)), " only conv 1x1 support"

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = padding if isinstance(
        padding, (tuple, list)) else (padding, padding, padding, padding)
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    # output shape
    out_height = (ih + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (iw + WPAD - dilated_kernel_w) // WSTR + 1
    pad_before = (0, 0, pad_top, pad_left, 0)
    pad_after = (0, 0, pad_down, pad_right, 0)
    # DOPAD
    DOPAD = HPAD != 0 or WPAD != 0
    if DOPAD and kernel_height > 1:
        data_pad = nn.pad(data, pad_before, pad_after, name="data_pad_3x3")
    else:
        data_pad = data
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    def get_factors_for_rf():
        for factor in range(4, 8):
            if (ic_chunk//factor)*factor == ic_chunk:
                return factor
        return 1
    rf_n= get_factors_for_rf()
    ic_rf_thread = (ic_chunk//rf_n)*ic_bn
    if ic_rf_thread != in_channel and use_rf:
        assert n==1,'only batch==1 is support'
        rf_oshape = (n*rf_n, oc_chunk, out_height, out_width, oc_bn)

        irf = te.reduce_axis((0, rf_n), name="irf")
        ic = te.reduce_axis((0, ic_rf_thread), name="ic")
        kh = te.reduce_axis((0, kernel_height), name="kh")
        kw = te.reduce_axis((0, kernel_width), name="kw")

        idxdiv = tvm.tir.indexdiv
        idxmod = tvm.tir.indexmod

        conv_out_s1 = te.compute(
            rf_oshape,
        lambda rf_n_, oc_chunk, oh, ow, oc_block: op_mad.mad(
            op_mad.mymul(
                data_pad[
                    rf_n_,
                    idxdiv(rf_n_*ic_rf_thread +ic, ic_bn),
                    oh * HSTR + kh * dilation_h,
                    ow * WSTR + kw * dilation_w,
                    idxmod(ic, ic_bn),
                ].astype(data.dtype)
                , kernel[rf_n_*ic_rf_thread +ic,oc_chunk, kh,
                        kw, 0, oc_block]
            ),
            axis=[ic, kh, kw],
            ),
            name="conv2d_NCHWc_s1",
            tag="conv2d_NCHWc",
        )
        #conv_out=conv_out_s1
        conv_out = te.compute(
            oshape,
            lambda n_, oc_chunk_, oh, ow, oc_block: te.sum(
                    conv_out_s1[
                        n_,
                        oc_chunk_,
                        oh,
                        ow//rf_n+irf,
                        oc_block,
                    ].astype(data.dtype),
                    axis=[irf],
                ),

            name="conv2d_NCHWc",
            tag="conv2d_NCHWc_rf",
        )
    else:
        ic = te.reduce_axis((0, in_channel), name="ic")
        kh = te.reduce_axis((0, kernel_height), name="kh")
        kw = te.reduce_axis((0, kernel_width), name="kw")

        idxdiv = tvm.tir.indexdiv
        idxmod = tvm.tir.indexmod

        conv_out = te.compute(
            oshape,
            lambda n, oc_chunk, oh, ow, oc_block: op_mad.mad(
                op_mad.mymul(
                    data_pad[
                        n,
                        idxdiv(ic, ic_bn),
                        oh * HSTR + kh * dilation_h,
                        ow * WSTR + kw * dilation_w,
                        idxmod(ic, ic_bn),
                    ].astype(data.dtype)
                    , kernel[ic,oc_chunk, kh,
                            kw, 0, oc_block]
                ),
                axis=[ic, kh, kw],
            ),
            name="conv2d_NCHWc",
            tag="conv2d_NCHWc",
        )
    conv_out.dtype = out_dtype
    flops = n * oc_chunk * out_height * out_width * oc_bn * kernel_height * kernel_width * ic_chunk * ic_bn * 2
    return conv_out, flops



@autotvm.register_topi_compute("conv2d_NCHWc_io.mali")
def conv2d_NCHWc_io(cfg, data, kernel, stride, padding, dilation, layout, out_layout, out_dtype):
    conv_out, flops = conv2d_NCHWc_io_op_common(data, kernel, stride, padding,
                                         dilation, layout, out_layout, True,
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
    #for 3x3 or 5x5 or 7x7 convolution
    kernel_max = max(kernel_height,kernel_width)
    def compute_at_axis_filter(y):
        #    if kernel_max > 3:
        #        return y.size[-1] in [3]
        #    elif kernel_max > 1:
        #        return y.size[-1] in [2]
        #    else:
        #        return y.size[-1] in [1]
        if kernel_height > 3:
            return y.size[-1] in [3]
        elif kernel_height > 1:
            return y.size[-1] in [3]
        else:
            return y.size[-1] in [1]

    #cfg.define_split("cmpat_when_kernel",
    #                 6,
    #                 num_outputs=2,
    #                 filter=compute_at_axis_filter)
    if kernel_max > 1:
        cfg.define_knob("auto_unroll_max_step", [0, 128, 256, 512])
        cfg.define_knob("unroll_explicit", [0, 1])
    # for mali======================
    
    #cfg.define_knob("idtype", [0, 1])
    #cfg.define_knob("kdtype", [0, 1])
    #cfg.define_knob("odtype", [0, 2])
    typedict = {0: "float32", 1: "climgfloatr32", 2: "climgfloatw32"}
    #data.dtype = typedict[cfg["idtype"].val]
    #kernel.dtype = typedict[cfg["kdtype"].val]
    #end define autotvm space
    cfg.add_flop(flops)
    #for mali=============
    #conv_out.dtype = typedict[cfg["odtype"].val]
    return conv_out


@autotvm.register_topi_schedule("conv2d_NCHWc.mali")
def schedule_conv2d_NCHWc(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(tensor):
        #op = tensor.op
        op = tensor
        if "conv2d_NCHWc" in op.tag:
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
            assert kh == 1 and kw == 1, "only support conv1x1 yet"
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

@autotvm.register_topi_schedule("conv2d_NCHWc_io.mali")
def schedule_conv2d_NCHWc_io(cfg, outs):
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


def conv2d_nchw_winograd_nchwc_conv3x3(data, kernel, strides, padding, dilation, out_dtype="float32"):
    assert False, "not support yet"
    #TODO
    """Compute conv2d internally using conv2d_nchwc layout for any dtype"""
    packed_out = conv2d_nchw_winograd_NCHWc_io(data, kernel, strides,
                              padding, dilation, "NCHW", "NCHW4c", out_dtype)
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def conv2d_nchw_winograd_nchwc_conv3x3_io(data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d internally using conv2d_nchwc layout for any dtype"""
    packed_out = conv2d_nchw_winograd_NCHWc_io(data, kernel, strides,
                              padding, dilation, "NCHW", "NCHW4c", out_dtype)

    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


@autotvm.register_topi_compute("conv2d_nchw_winograd_NCHWc_io.mali")
def conv2d_nchw_winograd_NCHWc_io(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
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
        CI, CO, KH, KW = get_const_tuple(kernel.shape)
        oc_bn = 4
        ic_bn = 4
        ic_chunk = CI // ic_bn
        oc_chunk = CO//oc_bn

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    pre_computed = False
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            dshape = (N, ic_chunk, IH, IW, ic_bn)
            data = tvm.te.placeholder(dshape, data.dtype, name="placeholder")
            kshape = (
                CI,
                oc_chunk,
                KH,
                KW,
                1,
                oc_bn,
            )
            kernel = tvm.te.placeholder(kshape,
                                        kernel.dtype,
                                        name="placeholder")
        else:
            data, kernel = _pack_data(data, kernel)
    else:
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

    # 'same' for convolution
    out_shape = (N, oc_chunk, H, W, oc_bn)
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
                     filter=lambda x: x.size[-1] == 4)
    cfg.define_split("data_hp", oc_chunk, num_outputs=3,
                     filter=lambda x: x.size[-1] == 4)

    cfg.define_split("bgemm_kp", oc_chunk, num_outputs=2, max_factor=32)
    cfg.define_split("bgemm_wp", oc_chunk, num_outputs=3,
                     filter=lambda x: x.size[-1] % 4 == 0)
    cfg.define_split("bgemm_hp", oc_chunk, num_outputs=2)
    ##### space definition end #####

    if pre_computed:
        U = kernel
    else:
        U = mali_winograd.kernel_transform(kernel, wino_size, G=G, out_dtype=out_dtype)
    V = mali_winograd.data_transform(data,
                                     wino_size,
                                     (pt, pl, pb, pr),
                                     B=B,
                                     out_dtype=out_dtype)
    M = mali_winograd.batch_gemm(U, V, out_dtype=out_dtype)
    output = mali_winograd.inverse_transform(
        out_shape, A=A, M=M, out_dtype=out_dtype)


    output.dtype = out_dtype
    #====================useless temperary===========begin
    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CO * H * W * KH * KW * CI)
    return output


@autotvm.register_topi_schedule("conv2d_nchw_winograd_NCHWc_io.mali")
def schedule_conv2d_nchw_winograd_NCHWc_io(cfg, outs):
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


@autotvm.register_topi_compute("conv2d_nchw_winograd.mali")
def conv2d_nchw_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype):
    tile_size = _pick_tile_size(data, kernel)
    return _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size)


@autotvm.register_topi_schedule("conv2d_nchw_winograd.mali")
def schedule_conv2d_nchw_winograd(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "winograd_conv2d_output" in op.tag:
            _schedule_winograd(cfg, s, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _decl_winograd(cfg, data, kernel, strides, padding, dilation, out_dtype, tile_size):
    N, CI, IH, IW = get_const_tuple(data.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    if len(kernel.shape) == 4:
        if dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))
        pre_computed = False
        CO, _, KH, KW = get_const_tuple(kernel.shape)
    else:
        assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"
        pre_computed = True
        H_CAT, W_CAT, CO, CI, VC = get_const_tuple(kernel.shape)
        CO *= VC
        KH, KW = H_CAT - tile_size + 1, W_CAT - tile_size + 1
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))

    assert KH == 3 and KW == 3 and HSTR == 1 and WSTR == 1
    data_pad = nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (IH + pt + pb - 3) // HSTR + 1
    W = (IW + pl + pr - 3) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW

    ##### space definition begin #####
    tile_bna_candidates = [1, 2, 4, 8, 16]
    factors = get_factors(CO)
    cfg.define_knob("tile_bna", [x for x in tile_bna_candidates if x in factors])
    cfg.define_knob("tile_bnb", [1, 2, 4, 8, 16])
    cfg.define_split("tile_t1", CI, num_outputs=2, max_factor=128)
    cfg.define_split("tile_t2", CO, num_outputs=2, max_factor=128)
    cfg.define_split("c_unroll", CI, num_outputs=2, max_factor=8)
    cfg.define_knob("yt", [1, 2, 4, 8, 16, 32])
    ##### space definition end #####

    if cfg.is_fallback:
        cfg["tile_bnb"].val = 4
        cfg["tile_bna"].val = 4
        while CO % cfg["tile_bna"].val != 0:
            cfg["tile_bna"].val //= 2
        cfg["yt"].val = 8
        cfg.fallback_split("tile_t1", [-1, 128])
        cfg.fallback_split("tile_t2", [-1, 128])
        cfg.fallback_split("c_unroll", [-1, 8])

    bna = cfg["tile_bna"].val
    bnb = cfg["tile_bnb"].val

    P_round = (P + bnb - 1) // bnb * bnb
    assert CO % bna == 0 and P_round % bnb == 0

    # pack input tile
    input_tile = te.compute(
        (CI, P_round // bnb, alpha, alpha, bnb),
        lambda ci, b, eps, nu, bb: tvm.tir.if_then_else(
            b * bnb + bb < P,
            data_pad[(b * bnb + bb) // (nH * nW)][ci][(b * bnb + bb) // nW % nH * m + eps][
                (b * bnb + bb) % nW * m + nu
            ],
            tvm.tir.const(0, data_pad.dtype),
        ),
        name="d",
    )
    #dont know why  cant use to tune this op when condition is True
    condition = True
    if autotvm.GLOBAL_SCOPE.in_tuning and condition:
        kvshape = (alpha, alpha, CO // bna, CI, bna)
        U = tvm.te.placeholder(kvshape, kernel.dtype, name="U")
    else:
        # transform kernel
        if pre_computed:
            U = kernel
        else:
            r_kh = te.reduce_axis((0, KH), "r_kh")
            r_kw = te.reduce_axis((0, KW), "r_kw")
            U = te.compute(
                (alpha, alpha, CO // bna, CI, bna),
                lambda eps, nu, co, ci, vco: te.sum(
                    kernel[co * bna + vco][ci][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw],
                    axis=[r_kh, r_kw],
                ),
                name="U",
            )

    # transform image
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    V = te.compute(
        (alpha, alpha, P_round // bnb, CI, bnb),
        lambda eps, nu, p, ci, vp: te.sum(
            input_tile[ci][p][r_a][r_b][vp] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="V",
    )

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    # batch gemm
    ci = te.reduce_axis((0, CI), name="c")
    M = te.compute(
        (alpha, alpha, CO, P_round),
        lambda eps, nu, co, p: te.sum(
            U[eps][nu][idxdiv(co, bna)][ci][idxmod(co, bna)]
            * V[eps][nu][idxdiv(p, bnb)][ci][idxmod(p, bnb)],
            axis=ci,
        ),
        name="M",
    )

    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    Y = te.compute(
        (CO, P, m, m),
        lambda co, p, vh, vw: te.sum(M[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]),
        name="Y",
    )

    # unpack output
    output = te.compute(
        (N, CO, H, W),
        lambda n, co, h, w: Y[
            co, n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m), idxmod(h, m), idxmod(w, m)
        ]
        # The following hack term is used to make the padding in batch gemm ("M")
        # effective, otherwise the padding will be eliminated by bound inference.
        # Use `tvm.tir.Mul` instead of `*` to avoid issues in const folding.
        + tvm.tir.Mul(tvm.tir.const(0, out_dtype), M[alpha - 1][alpha - 1][CO - 1][P_round - 1]),
        name="output",
        tag="winograd_conv2d_output",
    )

    # we have to manually assign effective GFLOP for winograd
    cfg.add_flop(2 * N * CO * H * W * KH * KW * CI)
    return output




def _schedule_winograd(cfg, s, op):
    """schedule winograd fast convolution F(2x2, 3x3) for conv2d"""
    # get ops and tensors
    output = op.output(0)

    Y = op.input_tensors[0]
    M, A = s[Y].op.input_tensors
    U, V = s[M].op.input_tensors
    d, B = s[V].op.input_tensors
    data_pad = s[d].op.input_tensors[0]

    # padding
    s[data_pad].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.te.ComputeOp):
        kernel, G = s[U].op.input_tensors
        s[G].compute_inline()
        (
            eps,
            nu,
            co,
            ci,
            vco,
        ) = s[U].op.axis
        #dont know why  cant use to tune this op when condition is False
        condition = False
        if not autotvm.GLOBAL_SCOPE.in_tuning or not condition:
            r_kh, r_kw = s[U].op.reduce_axis
            s[U].reorder(co, ci, eps, nu, r_kh, r_kw, vco)
            _ = [s[U].unroll(x) for x in [eps, nu, r_kh, r_kw]]
            s[U].vectorize(vco)
            tile_and_bind(s, U, co, ci, 1, 256)

        # dilation
        if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    # transform image
    s[B].compute_inline()
    VL = s.cache_write(V, "local")

    eps, nu, p, ci, vp = s[V].op.axis
    s[V].reorder(p, ci, eps, nu, vp)
    for axis in [eps, nu]:
        s[V].unroll(axis)
    s[V].vectorize(vp)
    fused = s[V].fuse(p, ci)

    bb, tt = cfg["tile_t1"].apply(s, V, fused)
    s[V].bind(bb, te.thread_axis("blockIdx.x"))
    s[V].bind(tt, te.thread_axis("threadIdx.x"))

    eps, nu, p, ci, vp = s[VL].op.axis
    r_a, r_b = s[VL].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        s[VL].unroll(axis)
    s[VL].vectorize(vp)
    s[d].compute_at(s[V], tt)
    s[VL].compute_at(s[V], tt)

    # batch gemm
    bna = cfg["tile_bna"].val
    bnb = cfg["tile_bnb"].val

    eps, nu, k, b = s[M].op.axis
    alpha = eps.dom.extent
    c = s[M].op.reduce_axis[0]
    yo, xo, yi, xi = s[M].tile(k, b, bna, bnb)
    c, c_unroll = cfg["c_unroll"].apply(s, M, c)
    s[M].reorder(yo, xo, c, c_unroll, yi, xi)
    s[M].unroll(c_unroll)
    s[M].unroll(yi)
    s[M].vectorize(xi)
    z = s[M].fuse(eps, nu)
    tile_and_bind3d(s, M, z, yo, xo, 1, cfg["yt"].val, 1)

    # inverse transform
    s[A].compute_inline()
    k, b, vh, vw = s[Y].op.axis
    r_a, r_b = s[Y].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[Y].unroll(axis)

    # schedule output and fusion
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0]

    n, co, h, w = s[output].op.axis
    m = alpha - 3 + 1
    h, w, hi, wi = s[output].tile(h, w, m, m)
    s[output].unroll(hi)
    s[output].unroll(wi)
    fused = s[output].fuse(n, co, h, w)
    bb, tt = cfg["tile_t2"].apply(s, output, fused)
    s[output].bind(bb, te.thread_axis("blockIdx.x"))
    s[output].bind(tt, te.thread_axis("threadIdx.x"))

    s[Y].compute_at(s[output], tt)

##### REGISTER ALTER OP LAYOUT #####
@nn.conv2d_alter_layout.register(["mali"])
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current

    new_attrs = {k: attrs[k] for k in attrs.keys()}

    strides = attrs.get_int_tuple("strides")
    padding = attrs.get_int_tuple("padding")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data, kernel = tinfos
    out_dtype = out_type.dtype

    impl, outs = relay.backend.compile_engine.select_implementation(
        relay.op.get("nn.conv2d"), attrs, tinfos, out_type, target
    )
    workload = autotvm.task.get_workload(outs)
    if workload is None:
        # The best implementation is not an AutoTVM template.
        # It may be from the auto-scheduler
        if impl.name.find("winograd") != -1:
            if dilation != (1, 1):
                logger.warning("Does not support weight pre-transform for dilated convolution.")
                return None

            assert data_layout == "NHWC" and kernel_layout == "HWIO"
            N, H, W, CI = get_const_tuple(data.shape)
            KH, KW, _, CO = get_const_tuple(kernel.shape)

            # Pre-compute weight transformation in winograd
            tile_size = _pick_tile_size(tinfos[0], tinfos[1], layout="NHWC")

            # HWIO -> OIHW
            kernel_transform = relay.transpose(inputs[1], axes=[3, 2, 0, 1])
            # alpha, alpha, CO, CI
            weight = relay.nn.contrib_conv2d_winograd_weight_transform(
                kernel_transform, tile_size=tile_size
            )
            new_attrs["tile_size"] = tile_size
            new_attrs["channels"] = CO
            return relay.nn.contrib_conv2d_winograd_without_weight_transform(
                inputs[0], weight, **new_attrs
            )

        return None
    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:  # if is fallback, clear query cache and return None
        autotvm.task.clear_fallback_cache(target, workload)
        return None

    topi_tmpl = workload[0]
    idxd = tvm.tir.indexdiv

    if topi_tmpl == "conv2d_nchw_spatial_pack.mali":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg["tile_co"].size[-1]

        new_attrs["kernel_layout"] = "OIHW%do" % VC

        new_data = data
        new_kernel = te.placeholder((idxd(CO, VC), CI, KH, KW, VC), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_spatial_pack.mali",
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.conv2d(*inputs, **new_attrs)
    if topi_tmpl == "conv2d_nchw_spatial_pack_io.mali":
        assert data_layout == "NCHW" and kernel_layout == "IOHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        VC = cfg["tile_co"].size[-1]

        new_attrs["kernel_layout"] = "IOHW%di" % VC
        new_data = data
        new_kernel = te.placeholder(
            (idxd(CI, VC), CO, KH, KW, VC), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_spatial_pack_io.mali",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)
    elif topi_tmpl == "conv2d_nchw_winograd.mali":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        tile_size = _pick_tile_size(data, kernel)
        VC = cfg["tile_bna"].val

        weight_expr = inputs[1]
        weight_expr = relay.nn.contrib_conv2d_winograd_weight_transform(
            weight_expr, tile_size=tile_size
        )
        weight_expr = relay.reshape(
            weight_expr, newshape=(KH + tile_size - 1, KW + tile_size - 1, idxd(CO, VC), VC, CI)
        )
        weight_expr = relay.transpose(weight_expr, axes=[0, 1, 2, 4, 3])

        new_attrs["tile_size"] = tile_size

        new_data = data
        new_kernel = te.placeholder(
            (KH + tile_size - 1, KW + tile_size - 1, idxd(CO, VC), CI, VC), kernel.dtype
        )
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd.mali",
        )
        dispatch_ctx.update(target, new_workload, cfg)

        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            inputs[0], weight_expr, **new_attrs
        )
    elif topi_tmpl == "conv2d_nchw_winograd_NCHWc_io.mali":
        assert data_layout == "NCHW" and kernel_layout == "IOHW"
        N, CI, H, W = get_const_tuple(data.shape)
        CO, _, KH, KW = get_const_tuple(kernel.shape)
        #tile_size = _pick_tile_size(data, kernel)
        tile_size = 2
        VC = 4

        data_expr = inputs[0]
        weight_expr = inputs[1]
        weight_expr = relay.nn.contrib_conv2d_winograd_weight_transform(
            weight_expr, tile_size=tile_size
        )
        #TODO results is not correct yet
        weight_expr = relay.reshape(weight_expr,
                                    newshape=(
                                        CI,
                                        idxd(CO, VC),
                                        KH + tile_size - 1,
                                        KW + tile_size - 1,
                                        1,
                                        VC,
                                    ))
        #weight_expr = relay.transpose(weight_expr, axes=[0, 1, 2, 4, 3])
        data_expr = relay.reshape(data_expr,
                                  newshape=(
                                      N,
                                      idxd(CI, VC),
                                      H,
                                      W,
                                      VC,
                                  ))

        new_attrs["tile_size"] = tile_size
        new_attrs["data_layout"] = "NCHW%dc" % VC
        new_attrs["kernel_layout"] = "IOHW1i%do" % VC

        new_data = te.placeholder((N, idxd(CI, VC), H, W, VC), data.dtype)
        new_kernel = te.placeholder((CI, idxd(CO, VC), KH + tile_size - 1,
                                    KW + tile_size - 1, 1, VC), kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, out_dtype],
            "conv2d_nchw_winograd_NCHWc_io.mali",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_winograd_without_weight_transform(
            data_expr, weight_expr, **new_attrs)
    elif topi_tmpl == "depthwise_conv2d_NCHWc_io.mali":
        # Converting NCHW to NCHWc.
        assert data_layout == "NCHW" and kernel_layout == "IOHW"
        data_tensor = data
        kernel_tensor = kernel

        batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
        channel_multiplier, out_channel, kh, kw = get_const_tuple(
            kernel_tensor.shape)
        ic_bn, oc_bn = 4, 4
        assert channel_multiplier == 1
        # update new attrs
        new_attrs["channels"] = out_channel
        new_attrs["data_layout"] = "NCHW%dc" % ic_bn
        new_attrs["kernel_layout"] = "IOHW1i%do" % oc_bn
        new_attrs["out_layout"] = "NCHW%dc" % ic_bn

        # Store altered operator's config.
        new_data = te.placeholder(
            (batch_size, in_channel // ic_bn, height, width, ic_bn), dtype=data.dtype
        )
        new_kernel = te.placeholder((1, out_channel // oc_bn, kh, kw, 1, oc_bn), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [
                new_data,
                new_kernel,
                strides,
                padding,
                dilation,
                new_attrs["data_layout"],
                new_attrs["out_layout"],
                out_dtype,
            ],
            topi_tmpl,
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_depthwise_conv2d_nchwc(*inputs, **new_attrs)
    elif topi_tmpl == "conv2d_NCHWc.intel_graphics":
        assert data_layout == "NCHW" and kernel_layout == "OIHW"
        batch_size, in_channel, height, width = get_const_tuple(
            data.shape)
        out_channel, _, kh, kw = get_const_tuple(kernel.shape)
        ic_bn = 4
        oc_bn = 4

        # update new attrs
        new_attrs["channels"] = out_channel
        new_attrs["data_layout"] = "NCHW%dc" % ic_bn
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs["kernel_layout"] = "OIHW%di%do" % (ic_bn, oc_bn)
        new_attrs["out_layout"] = "NCHW%dc" % oc_bn

        # Store altered operator's config
        new_data = te.placeholder(
            (batch_size, in_channel // ic_bn, height, width, ic_bn), dtype=out_type.dtype
        )
        new_kernel = te.placeholder(
            (out_channel // oc_bn, in_channel // ic_bn, kh, kw, ic_bn, oc_bn), dtype=kernel.dtype
        )
        new_workload = autotvm.task.args_to_workload(
            [
                new_data,
                new_kernel,
                strides,
                padding,
                dilation,
                new_attrs["data_layout"],
                new_attrs["out_layout"],
                out_dtype,
            ],
            "conv2d_NCHWc.intel_graphics",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)
    elif topi_tmpl == 'conv2d_NCHWc.mali':
        assert data_layout in ["NCHW"] and kernel_layout == "OIHW"
        ic_block_factor = oc_block_factor = 4
        N, CI, H, W = get_const_tuple(data.shape)
        new_layout = "NCHW4c"
        new_attrs["data_layout"] = new_layout
        new_attrs["out_layout"] = new_layout
        new_attrs["kernel_layout"] = "OIHW1o4i"
        CO, _, KH, KW = get_const_tuple(kernel.shape)

        new_attrs["channels"] = CO
        # Store the same config for the altered operator (workload)
        new_data = te.placeholder(
            (N, CI // ic_block_factor, H, W, ic_block_factor), dtype=data.dtype
        )
        new_kernel = te.placeholder(
            (
                CO,
                CI // oc_block_factor,
                KH,
                KW,
                1,
                oc_block_factor,
            ),
            dtype=kernel.dtype,
        )
        new_workload = autotvm.task.args_to_workload(
            [
                new_data,
                new_kernel,
                strides,
                padding,
                dilation,
                new_attrs["data_layout"],
                new_attrs["out_layout"],
                out_dtype,
            ],
            "conv2d_NCHWc.mali",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.conv2d(*inputs, **new_attrs)
    elif topi_tmpl == 'conv2d_NCHWc_io.mali':
        assert data_layout in ["NCHW"] and kernel_layout == "IOHW"
        ic_block_factor = oc_block_factor = 4
        N, CI, H, W = get_const_tuple(data.shape)
        new_layout = "NCHW4c"
        new_attrs["data_layout"] = new_layout
        new_attrs["out_layout"] = new_layout
        new_attrs["kernel_layout"] = "IOHW1i4o"
        _, CO, KH, KW = get_const_tuple(kernel.shape)


        new_attrs["channels"] = CO
        # Store the same config for the altered operator (workload)
        new_data = te.placeholder(
            (N, CI // ic_block_factor, H, W, ic_block_factor), dtype=data.dtype
        )
        new_kernel = te.placeholder(
            (
                CI,
                CO // oc_block_factor,
                KH,
                KW,
                1,
                oc_block_factor,
            ),
            dtype=kernel.dtype,
        )

        new_workload = autotvm.task.args_to_workload(
            [
                new_data,
                new_kernel,
                strides,
                padding,
                dilation,
                new_attrs["data_layout"],
                new_attrs["out_layout"],
                out_dtype,
            ],
            "conv2d_NCHWc_io.mali",
        )
        dispatch_ctx.update(target, new_workload, cfg)
        return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)
    else:
        return None


@conv2d_winograd_nhwc.register(["mali"])
def conv2d_winograd_nhwc_mali(
    data, weight, strides, padding, dilation, out_dtype, pre_computed=False
):
    """Conv2D Winograd in NHWC layout.
    This is a clean version to be used by the auto-scheduler for mali.
    """
    tile_size = _pick_tile_size(data, weight, layout="NHWC")
    return _conv2d_winograd_nhwc_impl(
        data, weight, strides, padding, dilation, out_dtype, tile_size, pre_computed
    )


##### SCHECULE UTILITIES #####
def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """ tile and bind to GPU threads """
    x_factor = x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, te.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, te.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi


def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, te.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, te.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, te.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, te.thread_axis("threadIdx.x"))
    s[tensor].reorder(zo, yo, xo, zi, yi, xi)
    return zo, yo, xo, zi, yi, xi
