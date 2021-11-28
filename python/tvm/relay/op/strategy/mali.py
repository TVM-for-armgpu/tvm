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
"""Definition of mali operator strategy."""
# pylint: disable=invalid-name,unused-argument,wildcard-import,unused-wildcard-import
import re
from tvm import topi
from tvm.auto_scheduler import is_auto_scheduler_enabled
from .generic import *
from .. import op as _op


@schedule_injective.register("mali")
def schedule_injective_mali(attrs, outs, target):
    """schedule injective ops for mali"""
    with target:
        return topi.mali.schedule_injective(outs)

# TCY TODO
@schedule_reduce.register("mali")
def schedule_reduce_mali(attrs, outs, target):
    """schedule reduction ops for mali"""
    with target:
        return topi.mali.schedule_reduce(outs)

# TCY TODO
@schedule_concatenate.register("mali")
def schedule_concatenate_mali(attrs, outs, target):
    """schedule concatenate for mali"""
    with target:
        return topi.mali.schedule_injective(outs)
        # return topi.mali.schedule_concatenate(outs)


@conv2d_strategy.register("mali")
def conv2d_strategy_mali(attrs, inputs, out_type, target):
    """conv2d mali strategy"""
    strategy = _op.OpStrategy()
    data, kernel = inputs
    dilation_h, dilation_w = attrs.get_int_tuple("dilation")
    stride_h, stride_w = attrs.get_int_tuple("strides")
    groups = attrs.groups
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    print(layout)
    print(kernel_layout)

    if dilation_h < 1 or dilation_w < 1:
        raise ValueError("dilation should be positive value")

    if groups == 1:
        if layout == "NCHW":
            if kernel_layout == "OIHW":
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.mali.conv2d_nchw_spatial_pack),
                    wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_spatial_pack),
                    name="conv2d_nchw_spatial_pack.mali",
                )
                # check if winograd algorithm is applicable
                _, _, kh, kw = get_const_tuple(kernel.shape)
                if (
                    kh == 3
                    and kw == 3
                    and stride_h == 1
                    and stride_w == 1
                    and dilation_h == 1
                    and dilation_w == 1
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.mali.conv2d_nchw_winograd),
                        wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_winograd),
                        name="conv2d_nchw_winograd.mali",
                        plevel=5,
                    )
                if (
                    kh == 1
                    and kw == 1
                    and dilation_h == 1
                    and dilation_w == 1
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(
                            topi.mali.conv2d_nchw_conv1x1),
                        wrap_topi_schedule(
                            topi.mali.schedule_conv2d_NCHWc),
                        name="conv2d_nchwc.mali",
                        plevel=15,
                    )

            elif kernel_layout == "IOHW":
                _, _, kh, kw = get_const_tuple(kernel.shape)
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.mali.conv2d_nchw_spatial_pack_io),
                    wrap_topi_schedule(
                        topi.mali.schedule_conv2d_nchw_spatial_pack_io),
                    name="conv2d_nchw_spatial_pack_io.mali",
                )
                if (
                    dilation_h == 1
                    and dilation_w == 1
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(
                            topi.mali.conv2d_nchw_conv1x1_io),
                        wrap_topi_schedule(
                            topi.mali.schedule_conv2d_NCHWc_io),
                        name="conv2d_nchwc_io.mali",
                        plevel=14,
                    )
                if (
                    kh == 3
                    and kw == 3
                    and stride_h == 1
                    and stride_w == 1
                    and dilation_h == 1
                    and dilation_w == 1
                ):
                    strategy.add_implementation(
                        wrap_compute_conv2d(
                            topi.mali.conv2d_nchw_winograd_nchwc_conv3x3_io),
                        wrap_topi_schedule(
                            topi.mali.schedule_conv2d_nchw_winograd_NCHWc_io),
                        name="conv2d_nchw_winograd_nchwc_io.mali",
                        plevel=15,
                    )
            elif re.match(r"OIHW\d*o", kernel_layout):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.mali.conv2d_nchw_spatial_pack),
                    wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_spatial_pack),
                    name="conv2d_nchw_spatial_pack.mali",
                )
            elif re.match(r"IOHW\d*i", kernel_layout):
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.mali.conv2d_nchw_spatial_pack_io),
                    wrap_topi_schedule(
                        topi.mali.schedule_conv2d_nchw_spatial_pack_io),
                    name="conv2d_nchw_spatial_pack_io.mali",
                )
            else:
                raise RuntimeError(
                    "Unsupported weight layout {} for conv2d NCHW".format(kernel_layout)
                )
        elif layout == "NHWC":
            assert kernel_layout == "HWIO"
            if not is_auto_scheduler_enabled():
                raise RuntimeError(
                    "conv2d NHWC layout is not enabled for mali without auto_scheduler."
                )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.conv2d_nhwc, need_auto_scheduler_layout=True),
                naive_schedule,
                name="conv2d_nhwc.mali",
            )
            is_winograd_applicable = False
            if len(kernel.shape) == 4:
                kernel_h, kernel_w, _, _ = get_const_tuple(kernel.shape)
                is_winograd_applicable = (
                    "float" in data.dtype
                    and "float" in kernel.dtype
                    and kernel_h == 3
                    and kernel_w == 3
                    and stride_h == 1
                    and stride_w == 1
                    and dilation_h == 1
                    and dilation_w == 1
                )
            if is_winograd_applicable:
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc),
                    naive_schedule,  # this implementation should never be picked by autotvm
                    name="conv2d_nhwc.winograd",
                    plevel=15,
                )
        elif layout == "NCHW4c":
            assert kernel_layout in ["OIHW1o4i", "IOHW1i4o"]
            _, _, kh, kw,_,_ = get_const_tuple(kernel.shape)
            if (
                dilation_h == 1
                and dilation_w == 1
            ):
                if kernel_layout == "IOHW1i4o":
                    strategy.add_implementation(
                        wrap_compute_conv2d(topi.mali.conv2d_NCHWc_io, True, True),
                        wrap_topi_schedule(topi.mali.schedule_conv2d_NCHWc_io),
                        name="conv2d_NCHWc_io.mali",
                        plevel=14,
                    )
                else:
                    strategy.add_implementation(
                        wrap_compute_conv2d(
                            topi.mali.conv2d_NCHWc, True, True),
                        wrap_topi_schedule(
                            topi.mali.schedule_conv2d_NCHWc),
                        name="conv2d_NCHWc.mali",
                        plevel=14,
                    )
            if (
                kh == 3
                and kw == 3
                and stride_h == 1
                and stride_w == 1
                and dilation_h == 1
                and dilation_w == 1
            ):
                if kernel_layout == "IOHW1i4o":
                    strategy.add_implementation(
                        wrap_compute_conv2d(
                            topi.mali.conv2d_nchw_winograd_NCHWc_io, True, True),
                        wrap_topi_schedule(
                            topi.mali.schedule_conv2d_nchw_winograd_NCHWc_io),
                        name="conv2d_nchw_winograd_NCHWC_io.mali",
                        plevel=20,
                    )
                else:
                    raise RuntimeError(
                        "Unsupported conv2d_nchw_winograd_nchwc layout {} for mali".format(layout))
        else:
            raise RuntimeError("Unsupported conv2d layout {} for mali".format(layout))
    elif is_depthwise_conv2d(data.shape, layout, kernel.shape, kernel_layout, groups):
        if layout == "NCHW":
            assert kernel_layout in ["IOHW", "OIHW"]
            if kernel_layout == "IOHW":
                channel_multiplier,__, _,_ = get_const_tuple(kernel.shape)
                assert kernel_layout == "IOHW"
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.mali.depthwise_conv2d_nchw_io),
                    wrap_topi_schedule(
                        topi.mali.schedule_depthwise_conv2d_nchw_io),
                    name="depthwise_conv2d_nchw_io.mali",
                )
                if channel_multiplier == 1:
                    strategy.add_implementation(
                        wrap_compute_conv2d(
                            topi.mali.depthwise_conv2d_NCHWc_io_wrap),
                        wrap_topi_schedule(
                            topi.mali.schedule_depthwise_conv2d_nchwc_io_wrap),
                        name="depthwise_conv2d_nchwc_io.mali",
                        plevel=15,
                    )
                else:
                    _ = "not support for the other channel_multiplier"
            elif layout == "NCHW4c":
                if channel_multiplier == 1:
                    strategy.add_implementation(
                        wrap_compute_conv2d(
                            topi.mali.depthwise_conv2d_NCHWc_io),
                        wrap_topi_schedule(
                            topi.mali.schedule_depthwise_conv2d_nchwc_io),
                        name="depthwise_conv2d_nchwc_io.mali",
                        plevel=15,
                    )
                else:
                    raise RuntimeError(
                        "Unsupported depthwise_conv2d layout {} for mali when channel_multiplier !=1".format(layout)
                        )
            else:
                assert kernel_layout == "OIHW"
                strategy.add_implementation(
                    wrap_compute_conv2d(topi.mali.depthwise_conv2d_nchw),
                    wrap_topi_schedule(topi.mali.schedule_depthwise_conv2d_nchw),
                    name="depthwise_conv2d_nchw.mali",
                )
        elif layout == "NHWC":
            assert kernel_layout == "HWOI"
            if not is_auto_scheduler_enabled():
                raise RuntimeError(
                    "depthwise_conv2d NHWC layout is not enabled for mali without auto_scheduler."
                )
            strategy.add_implementation(
                wrap_compute_conv2d(topi.nn.depthwise_conv2d_nhwc),
                naive_schedule,
                name="depthwise_conv2d_nhwc.mali",
            )
        else:
            raise RuntimeError("Unsupported depthwise_conv2d layout {} for mali".format(layout))
    else:  # group_conv2d
        raise RuntimeError("group_conv2d is not supported for mali")
    return strategy

@override_native_generic_func("conv2d_NCHWc_strategy")
def conv2d_NCHWc_strategy_mali(attrs, inputs, out_type, target):
    """conv2d_NCHWc mali strategy"""
    #logger.warning("conv2d_NCHWc can only be used for this platform.")
    kernel_layout = attrs.kernel_layout
    assert kernel_layout in ["IOHW1i4o", "OIHW1o4i"]
    strategy = _op.OpStrategy()
    if kernel_layout == "OIHW1o4i":
        strategy.add_implementation(
            wrap_compute_conv2d(
                topi.mali.conv2d_NCHWc,True,True),
            wrap_topi_schedule(
                topi.mali.schedule_conv2d_NCHWc),
            name="conv2d_NCHWc.mali",
            plevel=14,
        )
    else:
        strategy.add_implementation(
            wrap_compute_conv2d(
                topi.mali.conv2d_NCHWc_io, True, True),
            wrap_topi_schedule(
                topi.mali.schedule_conv2d_NCHWc_io),
            name="conv2d_NCHWc_io.mali",
            plevel=14,
        )
    return strategy

@depthwise_conv2d_NCHWc_strategy.register("mali")
def depthwise_conv2d_NCHWc_strategy_mali(attrs, inputs, out_type, target):
    """depthwise_conv2d_NCHWc adopted from arm_cpu"""
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        wrap_compute_conv2d(topi.mali.depthwise_conv2d_NCHWc_io, True, True),
        wrap_topi_schedule(topi.mali.schedule_depthwise_conv2d_NCHWc_io),
        name="depthwise_conv2d_NCHWc_io.mali",
        plevel=14,
    )
    return strategy


@conv2d_winograd_without_weight_transfrom_strategy.register("mali")
def conv2d_winograd_without_weight_transfrom_strategy_mali(attrs, inputs, out_type, target):
    """conv2d_winograd_without_weight_transfrom mali strategy"""
    dilation = attrs.get_int_tuple("dilation")
    groups = attrs.get_int("groups")
    layout = attrs.data_layout
    kernel_layout = attrs.kernel_layout
    strides = attrs.get_int_tuple("strides")
    kernel = inputs[1]
    assert dilation == (1, 1), "Do not support dilate now"
    assert strides == (1, 1), "Do not support strides now"
    assert groups == 1, "Do not supoort arbitrary group number"
    strategy = _op.OpStrategy()
    if layout == "NCHW":
        if kernel_layout == "IOHW":
            assert False, "TODO"
        else:
            assert len(kernel.shape) == 5, "Kernel must be packed into 5-dim"
            strategy.add_implementation(
                wrap_compute_conv2d(topi.mali.conv2d_nchw_winograd),
                wrap_topi_schedule(topi.mali.schedule_conv2d_nchw_winograd),
                name="conv2d_nchw_winograd.mali",
            )
    elif layout == "NCHW4c":
        assert len(
            kernel.shape) == 6, "Kernel must be packed into 6d-dim nchwc1i4o"
        strategy.add_implementation(
            wrap_compute_conv2d(topi.mali.conv2d_nchw_winograd_nchwc_io,True,True),
            wrap_topi_schedule(
                topi.mali.schedule_conv2d_nchw_winograd_NCHWc_io),
            name="conv2d_nchw_winograd_NCHWc_io.mali",
        )
    elif layout == "NHWC":
        if not is_auto_scheduler_enabled():
            raise RuntimeError(
                "Winograd conv2d NHWC is not enabled for mali without auto_scheduler."
            )
        strategy.add_implementation(
            wrap_compute_conv2d(topi.nn.conv2d_winograd_nhwc_without_weight_transform),
            naive_schedule,  # this implementation should never be picked by autotvm
            name="conv2d_nhwc_winograd_without_weight_transform",
            plevel=15,
        )

    else:
        raise RuntimeError(
            "Unsupported conv2d_winograd_without_weight_transfrom layout {}".format(layout)
        )
    return strategy


@dense_strategy.register("mali")
def dense_strategy_mali(attrs, inputs, out_type, target):
    """dense mali strategy"""
    strategy = _op.OpStrategy()
    if not is_auto_scheduler_enabled():
        strategy.add_implementation(
            wrap_compute_dense(topi.mali.dense_NW4w),
            wrap_topi_schedule(topi.mali.schedule_dense_NW4w),
            name="dense_NW4w.mali",
        )
    else:
        strategy.add_implementation(
            wrap_compute_dense(topi.nn.dense, need_auto_scheduler_layout=True),
            naive_schedule,
            name="dense.mali",
        )
    return strategy


@softmax_strategy.register("mali")
def softmax_strategy_mali(attrs, inputs, out_type, target):
   """softmax mali strategy"""
   strategy = _op.OpStrategy()
   strategy.add_implementation(
       wrap_compute_softmax(topi.nn.softmax),
       wrap_topi_schedule(topi.mali.schedule_softmax),
       name="softmax.mali",
   )
   return strategy


@schedule_pool.register("mali")
def schedule_pool_mali(attrs, outs, target):
    """schedule pooling ops for mali"""
    with target:
        return topi.mali.schedule_pool(outs, attrs.layout)


@schedule_pool_grad.register("mali")
def schedule_pool_grad_mali(attrs, outs, target):
    """schedule pooling gradient ops for mali"""
    with target:
        return topi.mali.schedule_pool_grad(outs)

@schedule_adaptive_pool.register("mali")
def schedule_adaptive_pool_mali(attrs, outs, target):
    """schedule adaptive pooling ops for mali"""
    with target:
        return topi.mali.schedule_adaptive_pool(outs, attrs.layout)
