"""Schedule for pooling operators"""
import tvm
from tvm import te
from tvm import autotvm
from .. import tag
from ..utils import traverse_inline


def schedule_adaptive_pool(outs, layout="NCHW"):
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

    def _schedule(Pool):
        num_thread = 8
        block_x = te.thread_axis("blockIdx.x")
        block_y = te.thread_axis("blockIdx.y")
        thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
        if Pool.op in s.outputs:
            Out = Pool
            OL = s.cache_write(Pool, "local")
        else:
            Out = outs[0].op.output(0)
            s[Pool].set_scope("local")
        if len(s[Out].op.axis) == 5:
            Pool_in = Pool.op.input_tensors[0]
            #Pool_inL = s.cache_read(Pool_in, "local", [Pool])
            n,cp,hp,wp,bcn4 = s[Out].op.axis
        else:
            n,cp,hp,wp = s[Out].op.axis
        by, ty = s[Out].split(n, factor=num_thread)
        if layout == "NHWC":
            bx, tx = s[Out].split(s[Out].op.axis[3], factor=num_thread)
        elif layout == "NCHW4c":
            bx, tx = s[Out].split(cp, factor=num_thread)
        else:
            bx, tx = s[Out].split(s[Out].op.axis[1], factor=num_thread)
        if len(s[Out].op.axis) == 5:
            s[Out].reorder(by, bx, ty, tx, hp, wp, bcn4)
        else:
            s[Out].reorder(by, bx, ty, tx)
        s[Out].bind(ty, thread_y)
        s[Out].bind(tx, thread_x)
        s[Out].bind(by, block_y)
        s[Out].bind(bx, block_x)
        if Pool.op in s.outputs:
            s[OL].compute_at(s[Out], tx)
            PoolL = OL
        else:
            s[Pool].compute_at(s[Out], hp)
            Pool.dtype = "float32"
            PoolL = Pool
        if len(s[Out].op.axis) == 5:
            _, cp, h, w, bn4 = s[PoolL].op.axis
            s[PoolL].reorder(cp, h, w, bn4)
            s[PoolL].vectorize(bn4)
            #_, cp, h, w, bn4 = s[Out].op.axis
            #s[Out].reorder(cp, h, w, bn4)
            s[Out].vectorize(bcn4)


    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp
                              ) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule global_pool
        elif OP.tag.startswith("adaptive_pool"):
            Pool = OP.output(0)
            _schedule(Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s

@autotvm.register_topi_compute("adaptive_pool.mali")
def adaptive_pool(_, data, out_dtype=None):
   a=0
   a=a+1
   return a

@autotvm.register_topi_schedule("adaptive_pool.mali")
def _schedule_adaptive_pool(_, data, out_dtype=None):
   a=0
   a=a+1
   return a