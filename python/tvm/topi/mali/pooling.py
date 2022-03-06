"""Schedule for pooling operators"""
import tvm
from tvm import te
from .. import tag
from ..utils import traverse_inline


def schedule_adaptive_pool(outs, layout):
    """Schedule for adaptive_pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of adaptive_pool
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for adaptive_pool.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    B = outs[0]
    if tag.is_broadcast(B.op.tag):
        TBUF = s[B].op.input_tensors[0]
        if (layout == "NCHW"):
            print("Pool NCHW")
            n, c, h, w = s[B].op.axis
            c, p4 = s[B].split(c, factor=4)
        else:
            print("Pool NCHW4C")
            n, c, h, w, p4 = s[B].op.axis
        cpo, cpi = s[B].split(c, factor=1)
        s[B].bind(cpo, te.thread_axis("blockIdx.x"))
        s[B].bind(cpi, te.thread_axis("threadIdx.x"))
        s[B].reorder(cpo, cpi, n)
        s[B].vectorize(p4)
        s[TBUF].compute_at(s[B], n)
        if (layout == "NCHW"):
            n, c, h, w = s[TBUF].op.axis
            c, p4 = s[TBUF].split(c, factor=4)
        else:
            n, c, h, w, p4 = s[TBUF].op.axis
        s[TBUF].vectorize(p4)
    elif B.op.tag.startswith("adaptive_pool"):
        if (layout == "NCHW"):
            n, c, h, w = s[B].op.axis
            c, p4 = s[B].split(c, factor=4)
        else:
            n, c, h, w, p4 = s[B].op.axis
        BL = s.cache_write(B, "local")
        cpo, cpi = s[B].split(c, factor=1)
        s[B].bind(cpo, te.thread_axis("blockIdx.x"))
        s[B].bind(cpi, te.thread_axis("threadIdx.x"))
        s[B].reorder(cpo, cpi, n)
        s[B].vectorize(p4)
        s[BL].compute_at(s[B], n)
        if (layout == "NCHW"):
            n, c, h, w = s[BL].op.axis
            c, p4 = s[BL].split(c, factor=4)
        else:
            n, c, h, w, p4 = s[BL].op.axis
        s[BL].reorder(c, h, w, n)
        s[BL].vectorize(p4)


    else:
        raise RuntimeError("Unsupported operator: %s" % B.op.tag)

    return s

def schedule_pool(outs, layout):
    """Schedule for pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of pool
        in the format of an array of tensors.

    layout: str
        Data layout.

    Returns
    -------
    s: Schedule
        The computation schedule for pool.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    print(layout)
    s = te.create_schedule(outs[0].op)

    B = outs[0]
    # inline all one-to-one-mapping operators except the last stage (output)
    if tag.is_broadcast(B.op.tag):    # avg - True, max - False
        TBUF = s[B].op.input_tensors[0]
        if (layout == "NCHW"):
            n, c, h, w = s[B].op.axis
            c, p4 = s[B].split(c, factor=4)
        else:
            n, c, h, w, p4 = s[B].op.axis
        cpo, cpi = s[B].split(c, factor=1)
        hpo, hpi = s[B].split(h, factor=1)
        wpo, wpi = s[B].split(w, factor=1)

        s[B].bind(cpo, te.thread_axis("blockIdx.z"))
        s[B].bind(cpi, te.thread_axis("threadIdx.z"))
        s[B].bind(hpi, te.thread_axis("blockIdx.y"))
        s[B].bind(hpo, te.thread_axis("threadIdx.y"))
        s[B].bind(wpi, te.thread_axis("blockIdx.x"))
        s[B].bind(wpo, te.thread_axis("threadIdx.x"))
        s[B].reorder(wpi, hpo, cpo, cpi, hpi, wpo, n)
        s[B].vectorize(p4)

        s[TBUF].compute_at(s[B], n)
        if (layout == "NCHW"):
            n, c, h, w = s[TBUF].op.axis
            c, p4 = s[TBUF].split(c, factor=4)
        else:
            n, c, h, w, p4 = s[TBUF].op.axis
        # dh, dw = s[TBUF].op.reduce_axis
        s[TBUF].reorder(c, h, w, n)
        s[TBUF].vectorize(p4)
        # s[TBUF].unroll(dh)
        # s[TBUF].unroll(dw)
    # schedule pool
    elif B.op.tag == 'pool_max':
        OL = s.cache_write(B, "local")
        n, c, h, w, p4 = s[B].op.axis
        cpo, cpi = s[B].split(c, factor=1)
        hpo, hpi = s[B].split(h, factor=1)
        wpo, wpi = s[B].split(w, factor=1)
        s[B].bind(cpo, te.thread_axis("blockIdx.z"))
        s[B].bind(cpi, te.thread_axis("threadIdx.z"))
        s[B].bind(hpi, te.thread_axis("blockIdx.y"))
        s[B].bind(hpo, te.thread_axis("threadIdx.y"))
        s[B].bind(wpi, te.thread_axis("blockIdx.x"))
        s[B].bind(wpo, te.thread_axis("threadIdx.x"))
        s[B].reorder(wpi, hpo, cpo, cpi, hpi, wpo, n)
        s[B].vectorize(p4)

        s[OL].compute_at(s[B], n)
        if (layout == "NCHW"):
            n, c, h, w = s[OL].op.axis
            c, p4 = s[OL].split(c, factor=4)
        else:
            n, c, h, w, p4 = s[OL].op.axis
        dh, dw = s[OL].op.reduce_axis
        s[OL].reorder(c, h, w, n)
        s[OL].vectorize(p4)
        s[OL].unroll(dh)
        s[OL].unroll(dw)
    else:
        raise RuntimeError("Unsupported operator: %s" % B.op.tag)

    return s

def schedule_pool_grad(outs):
    """Schedule for pool_grad on mali
    TCYTODO
    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of pool_grad
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for pool_grad.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule_pool_grad(op):
        if op in s.outputs:
            out = op
        else:
            out = outs[0].op.output(0)
        fused = s[out].fuse(*s[out].op.axis)
        num_thread = tvm.target.Target.current(allow_none=False).max_num_threads
        bx, tx = s[out].split(fused, factor=num_thread)
        s[out].bind(bx, te.thread_axis("blockIdx.x"))
        s[out].bind(tx, te.thread_axis("threadIdx.x"))

        if tag.COMM_REDUCE_IDX in op.input_tensors[0].op.tag:
            max_pool_index = op.input_tensors[0]
            s[max_pool_index].compute_at(s[out], tx)

            pool_input = max_pool_index.op.input_tensors[0]
            if isinstance(pool_input.op, tvm.te.ComputeOp):
                # handle padding
                s[pool_input].compute_inline()
        if op not in s.outputs:
            s[op].compute_at(s[out], tx)

    def _callback(op):
        if op.tag.startswith("pool_grad"):
            _schedule_pool_grad(op)

    traverse_inline(s, outs[0].op, _callback)

    return s
