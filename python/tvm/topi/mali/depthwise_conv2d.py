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
# pylint: disable=invalid-name,unused-variable,unused-argument
"""depthwise_conv2d schedule on ARM Mali GPU"""

import tvm
from tvm import te
from tvm import autotvm
import numpy as np

from .. import nn
from ..utils import traverse_inline
from ..nn.conv2d import unpack_NCHWc_to_nchw
from ..utils import get_const_tuple
from ..nn.utils import get_pad_tuple
from ..nn.depthwise_conv2d import _get_workload, depthwise_conv2d_infer_layout

# register original implementation of depthwise_conv2d_nchw since we don't need to change this part
@autotvm.register_topi_compute("depthwise_conv2d_nchw_io.mali")
def depthwise_conv2d_nchw_io(cfg, data, kernel, strides, padding, dilation, out_dtype):
    out = nn.depthwise_conv2d_nchw_iohw(data, kernel, strides, padding, dilation, out_dtype)
    shape = [int(i) for i in out.shape[:]+kernel.shape[2:]]
    cfg.add_flop(np.prod(shape))
    return out

# register customized schedule for arm cpu.
@autotvm.register_topi_schedule("depthwise_conv2d_nchw_io.mali")
def schedule_depthwise_conv2d_nchw_io(cfg, outs):
    """Schedule depthwise conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of depthwise convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(pad_data, kernel, conv):
        """schedule depthwise_conv2d"""
        max_unroll = 16
        vec_size = [1, 2, 4, 8, 16]

        ##### space definition begin #####
        n, c, y, x = s[conv].op.axis
        bc, tc, ci = cfg.define_split("tile_c", c, num_outputs=3)
        by, ty, yi = cfg.define_split("tile_y", y, num_outputs=3)
        bx, tx, xi = cfg.define_split("tile_x", x, num_outputs=3)
        cfg.define_annotate(
            "ann_spatial", [ci, yi, xi], policy="try_unroll_vec")

        # fallback support
        if cfg.is_fallback:
            ref_log = autotvm.tophub.load_reference_log(
                "mali", "rk3399", "depthwise_conv2d_nchw.mali"
            )
            cfg.fallback_with_reference_log(ref_log)
        ###### space definition end ######

        # schedule padding
        n, c, y, x = s[pad_data].op.axis
        tile_and_bind3d(s, pad_data, c, y, x, cfg["tile_c"].size[1], 1, 1)

        # schedule dilation
        if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

        # schedule conv
        if conv.op not in s.outputs:
            s[conv].set_scope("local")
            OL = conv
            output = s.outputs[0].output(0)
        else:
            OL = s.cache_write(conv, "local")
            output = conv

        n, c, y, x = s[output].op.axis
        bc, tc, ci = cfg["tile_c"].apply(s, output, c)
        by, ty, yi = cfg["tile_y"].apply(s, output, y)
        bx, tx, xi = cfg["tile_x"].apply(s, output, x)

        bc = s[output].fuse(n, bc)
        s[output].bind(bc, te.thread_axis("blockIdx.z"))
        s[output].bind(tc, te.thread_axis("threadIdx.z"))
        s[output].bind(by, te.thread_axis("blockIdx.y"))
        s[output].bind(ty, te.thread_axis("threadIdx.y"))
        s[output].bind(bx, te.thread_axis("blockIdx.x"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))

        di, dj = s[OL].op.reduce_axis
        s[OL].unroll(di)
        s[OL].unroll(dj)

        s[OL].compute_at(s[output], tx)
        n, ci, yi, xi = s[OL].op.axis

        cfg["ann_spatial"].apply(
            s,
            OL,
            [ci, yi, xi],
            axis_lens=[cfg["tile_c"].size[2],
                       cfg["tile_y"].size[2], cfg["tile_x"].size[2]],
            max_unroll=max_unroll,
            vec_size=vec_size,
            cfg=cfg,
        )

    def _callback(op):
        """traverse to find op to schedule"""
        # schedule depthwise_conv2d
        if op.tag == "depthwise_conv2d_nchw":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)
            _schedule(pad_data, kernel, conv)

    traverse_inline(s, outs[0].op, _callback)
    return s

# register original implementation of depthwise_conv2d_nchw since we don't need to change this part
@autotvm.register_topi_compute("depthwise_conv2d_nchw.mali")
def depthwise_conv2d_nchw(cfg, data, kernel, strides, padding, dilation, out_dtype):
    return nn.depthwise_conv2d_nchw(data, kernel, strides, padding, dilation, out_dtype)


def _pack_data(data, kernel,ic_bn, oc_bn):
    n, ic, ih, iw = get_const_tuple(data.shape)
    filters, cm, kh, kw = get_const_tuple(kernel.shape)
    oc = filters * cm
    ic_chunk = (ic+ic_bn-1) // ic_bn
    oc_chunk = (oc+oc_bn-1) // oc_bn

    data = te.compute(
        (n, ic_chunk, ih, iw, ic_bn),
        lambda bs, c, h, w, vc: data[bs, c * ic_bn + vc, h, w],
        name="data_vec",
        tag="data_pack",
    )

    kernel = te.compute(
        (1, oc_chunk, kh, kw, 1, oc_bn),
        lambda occ, icc, k_h, k_w, icb, ocb: kernel[
            (occ * oc_bn + ocb) // cm, (occ * oc_bn + ocb) % cm, k_h, k_w
        ],
        name="kernel_vec",
        tag="kernel_pack",
    )

    return data, kernel
# depthwise_conv2d_NCHWc implementation
def depthwise_conv2d_NCHWc_io_wrap(data, kernel, strides, padding, dilation, out_dtype):
    """Compute depthwise conv2d with NCHW layout."""
    layout = "NCHW"
    packed_out = depthwise_conv2d_NCHWc_io(
        data, kernel, strides, padding, dilation, layout, "NCHW4c", out_dtype
    )
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def schedule_depthwise_conv2d_nchwc_io_wrap(outs):
    """Create schedule for depthwise_conv2d_nchw."""
    return schedule_depthwise_conv2d_NCHWc_io(outs)


@autotvm.register_topi_compute("depthwise_conv2d_NCHWc_io.mali")
def depthwise_conv2d_NCHWc_io(
    cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype=None
):
    """Compute depthwise conv2d with NCHWc layout"""
    out_dtype = data.dtype if out_dtype is None else out_dtype

    if len(data.shape) == 5:
        batch, ic_chunk, in_height, in_width, ic_bn = get_const_tuple(data.shape)
        (
            cm_chunk,
            oc_chunk,
            filter_height,
            filter_width,
            cm_block,
            oc_bn,
        ) = get_const_tuple(kernel.shape)
        in_channel = ic_chunk * ic_bn
        out_channel = oc_chunk * oc_bn
        channel_multiplier = cm_chunk * cm_block
        assert channel_multiplier * in_channel == out_channel
    else:
        batch, in_channel, in_height, in_width = get_const_tuple(data.shape)
        channel_multiplier, out_channel, filter_height, filter_width = get_const_tuple(
            kernel.shape)
        ic_bn = oc_bn = 4
    assert channel_multiplier == 1

    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    HSTR, WSTR = strides

    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)

    dilated_kernel_h = (filter_height - 1) * dh + 1
    dilated_kernel_w = (filter_width - 1) * dw + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HPAD = pad_top + pad_down
    WPAD = pad_left + pad_right

    out_height = (in_height + HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (in_width + WPAD - dilated_kernel_w) // WSTR + 1

    cfg.define_knob("tile_ic", [4])
    cfg.define_knob("tile_oc", [4])
    # get workload and related schedule config
    wkl = _get_workload(
        te.placeholder((batch, in_channel, in_height, in_width), dtype=data.dtype),
        te.placeholder(
            (out_channel, channel_multiplier, filter_height, filter_width), dtype=kernel.dtype
        ),
        strides,
        (pad_top, pad_down),
        dilation,
        out_dtype,
    )
    cfg.is_fallback = False
    #if cfg.is_fallback:
    #_fallback_schedule(cfg, wkl)

    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            ic_bn = cfg["tile_ic"].val
            ic_chunk = in_channel // ic_bn
            oc_bn = cfg["tile_oc"].val
            oc_chunk = out_channel // oc_bn
            dshape = (batch, ic_chunk, in_height, in_width, ic_bn)
            data = tvm.te.placeholder(dshape, data.dtype, name="data")
            kshape = (1, oc_chunk, filter_height, filter_width, 1, oc_bn)
            kernel = tvm.te.placeholder(kshape, kernel.dtype, name="filter")
        else:
            data, kernel = _pack_data(data, kernel,ic_bn, oc_bn)
            _, _, _, _, ic_bn = get_const_tuple(data.shape)
            _, oc_chunk, _, _, _, oc_bn = get_const_tuple(kernel.shape)

    # padding stage
    DOPAD = pad_top != 0 or pad_left != 0 or pad_down != 0 or pad_right != 0
    if DOPAD:
        pad_before = [0, 0, pad_top, pad_left, 0]
        pad_after = [0, 0, pad_down, pad_right, 0]
        data_pad = nn.pad(data, pad_before, pad_after, name="PaddedInput")
    else:
        data_pad = data

    # depthconv stage
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    kh = te.reduce_axis((0, filter_height), name="kh")
    kw = te.reduce_axis((0, filter_width), name="kw")
    Output = te.compute(
        (batch, oc_chunk, out_height, out_width, oc_bn),
        lambda b, occk, oh, ow, ocb: te.sum(
            (
                data_pad[
                    b,
                    idxdiv(
                        idxdiv(occk * oc_bn + ocb, channel_multiplier), ic_bn
                    ),
                    oh * HSTR + kh * dh,
                    ow * WSTR + kw * dw,
                    idxmod(
                        idxdiv(occk * oc_bn + ocb, channel_multiplier), ic_bn
                    ),
                ].astype(out_dtype)
                * kernel[0, occk, kh, kw, 0, ocb].astype(out_dtype)
            ),
            axis=[kh, kw],
        ),
        name="DepthwiseConv2d",
        tag="depthwise_conv2d_NCHWc",
    )
    shape = [batch, oc_chunk, out_height, out_width, oc_bn,filter_height,filter_width]
    cfg.add_flop(np.prod(shape))
    return Output


@autotvm.register_topi_schedule("depthwise_conv2d_NCHWc_io.mali")
def schedule_depthwise_conv2d_NCHWc_io(cfg, outs):
    """CPU schedule for depthwise conv2d in NCHW[x]c layout"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        """Traverse operators from computation graph"""
        if "depthwise_conv2d_NCHWc" in op.tag:
            conv_out = op.output(0)
            data = conv_out.op.input_tensors[0]
            kernel = conv_out.op.input_tensors[1]
            _schedule_depthwise_conv2d_NCHWc_impl(s, cfg, data, kernel, conv_out, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_depthwise_conv2d_NCHWc_impl(s, cfg, pad_data, kernel, conv, op):
    """Schedule depthwise conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of depthwise convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    max_unroll = 16
    vec_size = [1, 2, 4, 8, 16]
    ##### space definition begin #####
    n, c, y, x,_ = s[conv].op.axis
    bc, tc = cfg.define_split("tile_cp", c, num_outputs=2)
    by, ty, yi = cfg.define_split("tile_hp", y, num_outputs=3)
    bx, tx, xi = cfg.define_split("tile_wp", x, num_outputs=3)#,filter=lambda y: y.size[-1] >=2)
    cfg.define_annotate(
        "ann_spatial", [yi, xi], policy="try_unroll_vec")

    # fallback support
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            "mali", "rk3399", "depthwise_conv2d_NCHWc_io.mali"
        )
        cfg.fallback_with_reference_log(ref_log)
    ###### space definition end ######

    # schedule pad and pack
    if isinstance(s[pad_data].op, tvm.te.ComputeOp) and "pad" in pad_data.op.tag:
        s[pad_data].compute_inline()
        data_pack, = s[pad_data].op.input_tensors
        if isinstance(s[data_pack].op, tvm.te.ComputeOp) and "pack" in data_pack.op.tag:
            s[data_pack].compute_inline()
    elif isinstance(s[pad_data].op, tvm.te.ComputeOp) and "pack" in pad_data.op.tag:
        s[pad_data].compute_inline()
    # kernel pack
    if isinstance(s[kernel].op, tvm.te.ComputeOp) and "pack" in kernel.op.tag:
        s[kernel].compute_inline()
    ## schedule dilation
    #if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
    #    s[kernel].compute_inline()
    output = conv
    #cache
    pad_dataL = s.cache_read(pad_data, "local", [conv])
    kernelL = s.cache_read(kernel, "local", [conv])
    OL = s.cache_write(conv, "local")

    # schedule conv
    #fuse relu add mul etc.
    if op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    len_4_flag = False
    if len(s[output].op.axis) == 4:
        len_4_flag = True
        n, cp, hp, wp = s[output].op.axis
    else:
        n, cp, hp, wp, Op4 = s[output].op.axis

    cpo, cpi = cfg["tile_cp"].apply(s, output, cp)
    wpo, wpi, Owp = cfg["tile_wp"].apply(s, output, wp)
    hpo, hpi, Ohp=cfg["tile_hp"].apply(s, output, hp)
    if len_4_flag:
        cpi, Op4 = s[output].split(cpi, factor=4)

    s[output].bind(cpo, te.thread_axis("blockIdx.z"))
    s[output].bind(cpi, te.thread_axis("threadIdx.z"))
    s[output].bind(hpo, te.thread_axis("blockIdx.y"))
    s[output].bind(hpi, te.thread_axis("threadIdx.y"))
    s[output].bind(wpo, te.thread_axis("blockIdx.x"))
    s[output].bind(wpi, te.thread_axis("threadIdx.x"))
    s[output].reorder(wpi, hpo, cpo, cpi, hpi, wpo, Ohp, n)

    s[OL].compute_at(s[output], n)

    k1, k2 = s[OL].op.reduce_axis
    _, _, hp, wp, p4 = s[OL].op.axis
    s[OL].reorder(hp, wp, p4)
    s[OL].vectorize(p4)
    #s[OL].unroll(wp)
    #s[OL].unroll(hp)
    #s[OL].unroll(k1)
    #s[OL].unroll(k2)

    #schedule UL VL local read
    s[pad_dataL].compute_at(s[OL], hp)
    s[kernelL].compute_at(s[OL], hp)

    #split Ul VL workload
    a, b, hp, wp, _, p4 = s[kernelL].op.axis
    s[kernelL].vectorize(p4)  # vectorize memory load
    #s[kernelL].unroll(wp)
    #s[kernelL].unroll(hp)
    _, _, hp, wp, p4 = s[pad_dataL].op.axis
    s[pad_dataL].vectorize(p4)  # vectorize memory load
    #s[pad_dataL].unroll(wp)
    #s[pad_dataL].unroll(hp)

    s[output].vectorize(Op4)  # vectorize memory load
    #s[output].unroll(Owp)  # vectorize memory load
    #s[output].unroll(Ohp)  # vectorize memory load

    #n, ci, yi, xi = s[OL].op.axis

    #cfg["ann_spatial"].apply(
    #    s,
    #    OL,
    #    [ci, yi, xi],
    #    axis_lens=[cfg["tile_c"].size[2],
    #                cfg["tile_y"].size[2], cfg["tile_x"].size[2]],
    #    max_unroll=max_unroll,
    #    vec_size=vec_size,
    #    cfg=cfg,
    #)




# register customized schedule for arm cpu.
@autotvm.register_topi_schedule("depthwise_conv2d_nchw.mali")
def schedule_depthwise_conv2d_nchw(cfg, outs):
    """Schedule depthwise conv2d

    Parameters
    ----------
    cfg: ConfigEntity
        The configuration of this template
    outs: Array of Tensor
        The computation graph description of depthwise convolution2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(pad_data, kernel, conv):
        """schedule depthwise_conv2d"""
        max_unroll = 16
        vec_size = [1, 2, 4, 8, 16]

        ##### space definition begin #####
        n, c, y, x = s[conv].op.axis
        bc, tc, ci = cfg.define_split("tile_c", c, num_outputs=3)
        by, ty, yi = cfg.define_split("tile_y", y, num_outputs=3)
        bx, tx, xi = cfg.define_split("tile_x", x, num_outputs=3)
        cfg.define_annotate("ann_spatial", [ci, yi, xi], policy="try_unroll_vec")

        # fallback support
        if cfg.is_fallback:
            ref_log = autotvm.tophub.load_reference_log(
                "mali", "rk3399", "depthwise_conv2d_nchw.mali"
            )
            cfg.fallback_with_reference_log(ref_log)
        ###### space definition end ######

        # schedule padding
        n, c, y, x = s[pad_data].op.axis
        tile_and_bind3d(s, pad_data, c, y, x, cfg["tile_c"].size[1], 1, 1)

        # schedule dilation
        if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

        # schedule conv
        if conv.op not in s.outputs:
            s[conv].set_scope("local")
            OL = conv
            output = s.outputs[0].output(0)
        else:
            OL = s.cache_write(conv, "local")
            output = conv

        n, c, y, x = s[output].op.axis
        bc, tc, ci = cfg["tile_c"].apply(s, output, c)
        by, ty, yi = cfg["tile_y"].apply(s, output, y)
        bx, tx, xi = cfg["tile_x"].apply(s, output, x)

        bc = s[output].fuse(n, bc)
        s[output].bind(bc, te.thread_axis("blockIdx.z"))
        s[output].bind(tc, te.thread_axis("threadIdx.z"))
        s[output].bind(by, te.thread_axis("blockIdx.y"))
        s[output].bind(ty, te.thread_axis("threadIdx.y"))
        s[output].bind(bx, te.thread_axis("blockIdx.x"))
        s[output].bind(tx, te.thread_axis("threadIdx.x"))

        di, dj = s[OL].op.reduce_axis
        s[OL].unroll(di)
        s[OL].unroll(dj)

        s[OL].compute_at(s[output], tx)
        n, ci, yi, xi = s[OL].op.axis

        cfg["ann_spatial"].apply(
            s,
            OL,
            [ci, yi, xi],
            axis_lens=[cfg["tile_c"].size[2], cfg["tile_y"].size[2], cfg["tile_x"].size[2]],
            max_unroll=max_unroll,
            vec_size=vec_size,
            cfg=cfg,
        )

    def _callback(op):
        """traverse to find op to schedule"""
        # schedule depthwise_conv2d
        if op.tag == "depthwise_conv2d_nchw":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            conv = op.output(0)
            _schedule(pad_data, kernel, conv)

    traverse_inline(s, outs[0].op, _callback)
    return s


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
    return zo, zi, yo, yi, xo, xi


@depthwise_conv2d_infer_layout.register("mali")
def _depthwise_conv2d_infer_layout(workload, cfg):
    """Infer input/output shapes and layouts from a workload and cfg.

    Parameters
    ----------
    workload : tuple
        conv2d workload

    cfg : tuple
        tvm.autotvm config

    Returns
    -------
    Output : [tuple of tuple and str, tuple of tuple and str]
        Input shapes and layouts, and output shapes and layouts
    """
    _, data, kernel, strides, padding, dilation, _, _, dtype = workload
    batch_size, in_channel, in_height, in_width = data[1]
    filter_channel, channel_multiplier, k_height, k_width = kernel[1]
    out_channel = filter_channel * channel_multiplier
    out_height = (in_height + padding[0] + padding[2] -
                  k_height) // strides[0] + 1
    out_width = (in_width + padding[1] + padding[3] -
                 k_width) // strides[1] + 1
    tile_ic, tile_oc = cfg["tile_ic"].val, cfg["tile_oc"].val
    in_shape = (batch_size, in_channel // tile_ic, in_height, in_width,
                tile_ic)
    in_layout = "NCHW%dc" % tile_ic
    out_shape = (batch_size, out_channel // tile_oc, out_height, out_width,
                 tile_oc)
    out_layout = "NCHW%dc" % tile_oc
    return ((in_shape, in_layout), ), ((out_shape, out_layout), )
