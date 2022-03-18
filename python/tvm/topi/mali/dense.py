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
# pylint: disable=invalid-name,unused-variable
"""dense schedule on ARM Mali GPU"""
from tvm import te
from tvm import autotvm

from .. import nn
from ..utils import traverse_inline
from .. import tag


@autotvm.register_topi_compute("dense.mali")
def dense(_, data, weight, bias=None, out_dtype=None):
    return nn.dense(data, weight, bias, out_dtype)


@autotvm.register_topi_compute("dense_NW4w.mali")
def dense_NW4w(_, data, weight, bias=None, out_dtype=None):
    """Dense operator on Mali"""
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.te.Tensor
        2-D with shape [out_dim, in_dim]

    bias : Optional[tvm.te.Tensor]
        1-D with shape [out_dim]

    out_dtype : Optional[str]
        The output type. This is used for mixed precision.

    auto_scheduler_rewritten_layout: str = ""
        The layout after auto-scheduler's layout rewrite pass.

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2, "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    auto_scheduler_rewritten_layout=''
    if auto_scheduler_rewritten_layout:
        # Infer shape for the rewritten layout
        out_dim, red_dim = auto_scheduler.get_shape_from_rewritten_layout(
            auto_scheduler_rewritten_layout, ["j", "k"]
        )
        auto_scheduler.remove_index_check(weight)
    else:
        out_dim, red_dim = weight.shape
    assert in_dim == red_dim
    oc_bn=4

    k = te.reduce_axis((0, in_dim), name="k")
    matmul = te.compute((batch, out_dim // oc_bn, oc_bn),
                        lambda x, y, bn: te.sum(
                            data[x, k] * weight[y * oc_bn + bn, k], axis=k),
                        name="T_dense",
                        tag="dense",
                        attrs={"layout_free_placeholders": [weight]})
    if bias is not None:
        matmul = te.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j // 4, j % 4] + bias[j].astype(out_dtype),
            attrs={"data_type": "buffer"},
            name="T_bias_add",
            tag=tag.BROADCAST,
        )
    else:
        matmul = te.compute((batch, out_dim),
                            lambda x, y: matmul[x, y // 4, y % 4],
                            attrs={"data_type": "buffer"},
                            tag=tag.ELEMWISE,
                            name="T_dense_unpack")
    if auto_scheduler_rewritten_layout:
        matmul = auto_scheduler.rewrite_compute_body(matmul, auto_scheduler_rewritten_layout)

    return matmul


@autotvm.register_topi_schedule("dense_NW4w.mali")
def schedule_dense_NW4w(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense":
            vec_size = [1, 2, 4, 8, 16]
            max_unroll = 32
            dense_out = op.output(0)
            output = outs[0]

            B = dense_out
            if B.op in s.outputs:
                A, W = s[B].op.input_tensors
            else:
                A, W = s[dense_out].op.input_tensors

            AL = s.cache_read(A, "local", [B])
            WL = s.cache_read(W, "local", [B])
            BL = s.cache_write(B, "local")

            if op not in s.outputs:
                s[B].compute_inline()
                B = s.outputs[0].output(0)
                if op.output(0).op not in s.outputs:
                    # sometimes there is only one element in output.op.input_tensors
                    Bp = output.op.input_tensors[0]
                    s[Bp].compute_inline()
                    B = s.outputs[0].output(0)

            if len(s[B].op.axis) == 3:
                hp, wp, Bwp4 = s[B].op.axis
                filter_f = lambda x: x.size[-1] > 0
            else:
                hp, wp = s[B].op.axis
                filter_f = lambda x: x.size[-1] % 4 == 0
            (c, ) = s[BL].op.reduce_axis
            ##### space definition begin #####
            cfg.define_split("tile_y", hp, num_outputs=2)
            cfg.define_split("tile_x", wp, num_outputs=2, filter = filter_f)
            cfg.define_split("c_unroll", c, num_outputs=2, max_factor=64)
            ##### space definition end #####

            #wpo,wpi = s[B].split(wp, factor=32)
            wpo,wpi = cfg["tile_x"].apply(s, B, wp)
            hpo,hpi = cfg["tile_y"].apply(s, B, hp)
            if len(s[B].op.axis) == 2:
                wpi, Bwp4 = s[B].split(wpi, factor=4)

            s[B].bind(wpo, te.thread_axis("blockIdx.x"))
            s[B].bind(wpi, te.thread_axis("threadIdx.x"))
            s[B].bind(hpo, te.thread_axis("blockIdx.y"))
            s[B].bind(hpi, te.thread_axis("threadIdx.y"))

            s[B].reorder(wpo, wpi, Bwp4, hpo, hpi)
            s[BL].compute_at(s[B], hpi)
            s[B].reorder(wpo, wpi, hpo, hpi, Bwp4)

            if len(s[BL].op.axis) == 3:
                hp, wp,wp_p4 = s[BL].op.axis
            else:
                hp, wp = s[BL].op.axis
                wpo, wp_p4 = s[BL].split(wp, factor=4)

            (k,) = s[BL].op.reduce_axis

            #ko, ki = s[BL].split(k, nparts=32)
            ko, ki = cfg["c_unroll"].apply(s, BL, k)

            s[BL].reorder(ko, ki, wp_p4)
            s[BL].vectorize(wp_p4)
            s[BL].unroll(ki)

            s[WL].compute_at(s[BL], ko)
            wpo, wp = s[WL].op.axis
            s[WL].unroll(wpo)
            wpo, wp_p4 = s[WL].split(wp, factor=4)
            s[WL].vectorize(wp_p4)
            s[WL].unroll(wpo)

            s[AL].compute_at(s[BL], ko)
            _, wp = s[AL].op.axis
            wpo, wp_p4 = s[AL].split(wp, factor=4)
            s[AL].vectorize(wp_p4)
            s[AL].unroll(wpo)

            s[B].vectorize(Bwp4)
            B_unp=s.outputs[0].output(0)
    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_schedule("dense.mali")
def schedule_dense(cfg, outs):
    """Schedule for dense operator.

    Parameters
    ----------
    cfg: ConfigEntity
        The config entity for this template
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense":
            vec_size = [1, 2, 4, 8, 16]
            max_unroll = 32

            dense_out = op.output(0)
            output = outs[0]

            y, x = s[output].op.axis
            c = s[dense_out].op.reduce_axis[0]

            ##### space definition begin #####
            cfg.define_split("tile_y", y, num_outputs=3)
            cfg.define_split("tile_x", x, num_outputs=3)
            cfg.define_split("c_unroll", c, num_outputs=2, max_factor=64)

            # fallback support
            if cfg.is_fallback:
                ref_log = autotvm.tophub.load_reference_log("mali", "rk3399", "dense.mali")
                cfg.fallback_with_reference_log(ref_log)
            ##### space definition end #####

            if dense_out.op in s.outputs:
                dense_out = s.cache_write(output, "local")

            by, ty, yi = cfg["tile_y"].apply(s, output, y)
            bx, tx, xi = cfg["tile_x"].apply(s, output, x)

            s[output].bind(by, te.thread_axis("blockIdx.y"))
            s[output].bind(bx, te.thread_axis("blockIdx.x"))
            s[output].bind(ty, te.thread_axis("threadIdx.y"))
            s[output].bind(tx, te.thread_axis("threadIdx.x"))

            if cfg["tile_y"].size[-1] < max_unroll:
                s[output].unroll(yi)
            if cfg["tile_x"].size[-1] in vec_size:
                s[output].vectorize(xi)
            s[dense_out].compute_at(s[output], tx)

            k = s[dense_out].op.reduce_axis[0]
            y, x = s[dense_out].op.axis
            k, k_unroll = cfg["c_unroll"].apply(s, dense_out, k)
            s[dense_out].reorder(k, k_unroll, y, x)
            s[dense_out].unroll(k_unroll)
            if cfg["tile_y"].size[-1] < max_unroll:
                s[dense_out].unroll(y)
            if cfg["tile_x"].size[-1] in vec_size:
                s[dense_out].vectorize(x)

    traverse_inline(s, outs[0].op, _callback)
    return s


def fuse_and_bind(s, tensor, axis=None, num_thread=None):
    """ fuse all the axis and bind to GPU threads """
    # TODO(@comaniac): figure out where this function is used.
    axis = axis or s[tensor].op.axis
    fused = s[tensor].fuse(*axis)
    bx, tx = s[tensor].split(fused, num_thread)
    s[tensor].bind(bx, te.thread_axis("blockIdx.x"))
    s[tensor].bind(tx, te.thread_axis("threadIdx.x"))
    return bx, tx
