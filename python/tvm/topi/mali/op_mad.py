import tvm
import numpy
from tvm import te
import numpy as np
def mymul(x,y):
    """customized log intrinsic function"""
    return tvm.tir.call_intrin(x.dtype, "tir.mymul", x,y)

def mymad(x,y):
    """customized log intrinsic function"""
    return tvm.tir.call_intrin(x.dtype, "tir.mymad", y,x)

def read_imagef(data,x,y):
    """customized log intrinsic function"""
    return tvm.tir.call_intrin(x.dtype, "tir.read_imagef",data, x,y)

def write_imagef(data,x,y):
    """customized log intrinsic function"""
    return tvm.tir.call_intrin(x.dtype, "tir.write_imagef", data,x,y)

def my_opencl_readimagef_rule(op):
    """CUDA lowering rule for log"""
    return tvm.tir.call_pure_extern("float32", "read_imagef", op.args[0],op.args[1],op.args[2])

def my_opencl_writeimagef_rule(op):
    """CUDA lowering rule for log"""
    return tvm.tir.call_pure_extern("float32", "write_imagef", op.args[0],op.args[1],op.args[2])

def my_opencl_mymul_rule(op):
    """CUDA lowering rule for log"""
    return tvm.tir.call_pure_extern("float32", "mul_must_be_replace", op.args[0],op.args[1])

def my_opencl_mymad_rule(op):
    """CUDA lowering rule for log"""
    return tvm.tir.call_pure_extern("float32", "mad", op.args[0],op.args[1])


# new op registration is triggered by registering an attribute of the op
tvm.ir.register_op_attr("tir.mymul", "TCallEffectKind", tvm.tir.CallEffectKind.Pure)
tvm.ir.register_op_attr("tir.mymul", "TVectorizable",True)
tvm.target.register_intrin_rule("opencl", "mymul", my_opencl_mymul_rule, override=True)

tvm.ir.register_op_attr("tir.mymad", "TCallEffectKind", tvm.tir.CallEffectKind.Pure)
tvm.ir.register_op_attr("tir.mymad","TVectorizable",True)
tvm.target.register_intrin_rule("opencl", "mymad", my_opencl_mymad_rule, override=True)

tvm.ir.register_op_attr("tir.read_imagef", "TCallEffectKind", tvm.tir.CallEffectKind.Pure)
tvm.target.register_intrin_rule("opencl", "read_imagef", my_opencl_readimagef_rule, override=True)

tvm.ir.register_op_attr("tir.write_imagef", "TCallEffectKind", tvm.tir.CallEffectKind.Pure)
tvm.target.register_intrin_rule("opencl", "write_imagef", my_opencl_writeimagef_rule, override=True)

n = te.var("n")
m = te.var("m")
mad = te.comm_reducer(lambda x, y: mymad(x,y),
    lambda t: tvm.tir.const(0, dtype=t), name="mad")
print("registor op success")


