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
# pylint: disable=invalid-name, unused-variable, trailing-whitespace
"""Schedule for softmax operator"""
from tvm.target import Target
from tvm import te
from .. import generic


def schedule_softmax(outs):
    """Schedule for softmax op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    softmax = outs[0]
    tgt = Target.current(allow_none=False)

    op_tag = softmax.op.tag
    if op_tag == "softmax_output":
        expsum = softmax.op.input_tensors[1]
        exp = softmax.op.input_tensors[0]
        max_elem = s[exp].op.input_tensors[1]
    elif op_tag == "log_softmax_output":
        exp = None
        max_elem = softmax.op.input_tensors[1]
        expsum = softmax.op.input_tensors[2]
    else:
        raise ValueError(
            "Tag is expected to be softmax_output or log_softmax_output. \
                         Got {0}"                                                                                                                                                                     .format(
                op_tag
            )
        )

    # The nvptx and rocm backends only supports 32-bits warp shuffle
    # instructions.

    if len(softmax.shape) > 2:
        ops = [max_elem.op, expsum.op, softmax.op]
        if exp is not None:
            ops.append(exp.op)

        for op in ops:
            s = schedule_injective_from_existing(s, op.output(0))
    else:
        expsum = softmax.op.input_tensors[1]
        exp = softmax.op.input_tensors[0]
        max_elem = s[exp].op.input_tensors[1]
        num_thread = 64


        block_x = te.thread_axis("blockIdx.x")
        block_y = te.thread_axis("blockIdx.y")
        thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = te.thread_axis((0, num_thread), "threadIdx.y")

        #====================== exp
        if exp is not None:
            max_elemL = s.cache_read(max_elem, "local", [exp])
            ew, wh = exp.op.axis
            s[max_elemL].compute_at(s[exp], ew)
            s[exp].bind(ew, block_x)
            wh, wh4 = s[exp].split(wh, factor=4)
            s[exp].vectorize(wh4)
            who, whi = s[exp].split(wh, factor=num_thread)
            s[exp].bind(who, block_y)
            s[exp].bind(whi, thread_y)

        #====================== max_elem
        A = s[max_elem].op.input_tensors[0]
        AL = s.cache_read(A, "local", [max_elem])
        max_elemL = s.cache_write(max_elem, "local")
        meh, = max_elem.op.axis

        s[max_elem].bind(meh, block_x)
        s[max_elemL].compute_at(s[max_elem], meh)
        k, = s[max_elemL].op.reduce_axis
        ko, ki = s[max_elemL].split(k, nparts=8)
        s[max_elemL].unroll(ki)
        s[AL].compute_at(s[max_elemL], ko)
        alh, alw = s[AL].op.axis
        alw, alwp4 = s[AL].split(alw, factor=4)
        s[AL].vectorize(alwp4)
        s[AL].unroll(alw)

        #====================== expsum
        k = expsum.op.reduce_axis[0]
        ko, ki = s[expsum].split(k, factor=num_thread)

        #====================== softmax_norm
        expsumL = s.cache_read(expsum, "local", [softmax])
        softmaxL = s.cache_write(softmax, "local")
        tx, Sxi = s[softmax].split(softmax.op.axis[1], nparts=num_thread)
        s[softmax].bind(softmax.op.axis[0], block_x)
        s[softmax].bind(tx, thread_x)

        s[softmaxL].compute_at(s[softmax], tx)
        tx, xi = s[softmaxL].op.axis
        xio, xii = s[softmaxL].split(xi, factor=4)
        s[softmaxL].vectorize(xii)
        s[softmaxL].unroll(xio)

        s[expsumL].compute_at(s[softmaxL], tx)

        xio, xii = s[softmax].split(Sxi, factor=4)
        s[softmax].vectorize(xii)
        s[softmax].unroll(xio)
        #======================
    return s
