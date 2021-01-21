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
"""Testcode for Android RPC.

To use it, start an RPC tracker with "python -m tvm.exec.rpc_tracker".
Use the tracker's address and port when configuring the RPC app.
Use "android" as the key if you wish to avoid modifying this script.
"""

import tvm
from tvm import te
import os
from tvm import rpc
from tvm.contrib import utils, ndk
import numpy as np
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import extern_op

# Set to be address of tvm proxy.
tracker_host = '127.0.0.1'
tracker_port = 9090
key = "android"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch

# whether enable to execute test on OpenCL target
test_opencl = True 
# whether enable to execute test on Vulkan target
test_vulkan = False


def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"
    
    #cfg = autotvm.get_config()

    #cfg.define_knob("idtype",[0,1])
    #cfg.define_knob("kdtype",[0,1])
    #cfg.define_knob("wdtype",[0,2])
    typedict = {0:"float32",1:"climgfloatr32",2:"climgfloatw32"}

    # Algorithm
    ddtype = "climgfloatr32"
    batch = N
    in_channel = CI
    out_channel = CO
    in_size = H
    kernel = KW
    pad = padding[0]
    stride = stride[0]


    open_image=1
    ddtype='float32'
    if open_image == 1:
        ddtype = "climgfloatr32"
    
    # Algorithm
    PACK4 = 4
    #PACK2=2
    PACK2=1
    W_P = in_size//PACK2
    H_P = in_size//PACK2
    C_P = in_channel//PACK4
    K_P=out_channel//PACK4
    
    A = te.placeholder((C_P,H_P,W_P*PACK4), dtype=ddtype,name="A")
    W = te.placeholder((in_channel,out_channel), dtype=ddtype, name="W")
    out_size = in_size
    # Pad input
    Apad = A
    # Create reduction variables
    rc = te.reduce_axis((0, in_channel), name="rc")
    # Compute the convolution
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    B = te.compute(
        (K_P, H_P,W_P*PACK4),
        #lambda ff,yy,xx_p4: extern_op.mysum(
        lambda ff,yy,xx_p4: te.sum(
            Apad[idxdiv(rc,4),yy,idxmod(rc,4)+idxdiv(xx_p4,4)*4] * W[rc,ff*PACK4+idxmod(xx_p4,4)], axis=[rc]
            #extern_op.mymul(Apad[idxdiv(rc,4),yy,idxmod(rc,4)+idxdiv(xx_p4,4)*4] , W[rc,ff*PACK4+idxmod(xx_p4,4)]), axis=[rc]
        ),
        name="B",
    )
    if open_image ==1:
        B.dtype="climgfloatw32"
    
    # Designate the memory hierarchy
    s = te.create_schedule(B.op)
    #s[Apad].compute_inline()  # compute Apad inline
    #AA = s.cache_read(Apad, "shared", [B])
    #WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(Apad, "local", [B])
    WL = s.cache_read(W, "local", [B])
    BL = s.cache_write(B, "local")
    
    # tile consts
    tile = 1
    num_thread = 4
    block_factor = tile * num_thread
    step = 4
    vthread = 2
    
    # Get the GPU thread indices
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis( "threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")
    thread_xz = te.thread_axis("vthread", name="vx")
    thread_yz = te.thread_axis("vthread", name="vy")
    
    # Split the workloads
    kp, hp, wp_p4  = s[B].op.axis
    wp,wp4=s[B].split(wp_p4,factor=4*2)
    hpo,hpii=s[B].split(hp,factor=2)
    
    
    ########
    hpo,hpi=s[B].split(hpo,factor=block_factor)
    wpo,wpi=s[B].split(wp,factor=block_factor//2)
    kpo,kpi=s[B].split(kp,factor=block_factor*16)
    
    #target = 'llvm -mcpu=core-avx2'
    #func = tvm.build(s, [A, W, B], target=target, name='mmult')
    #np.random.seed(5)
    #a_np = np.random.uniform(size=(C_P,H_P,W_P,PACK4)).astype("float32")
    #w_np = np.random.uniform(size=(in_channel,out_channel)).astype("float32")
    #ctx = tvm.context(target, 0)
    #a = tvm.nd.array(a_np, ctx,dtype=A.dtype)
    #w = tvm.nd.array(w_np, ctx,dtype=W.dtype)
    #b = tvm.nd.array(np.zeros((K_P, H_P,W_P,PACK4), dtype="float32"), ctx, dtype=B.dtype)
    #func(a, w, b)
    #exit(0)
    #
    # Bind the iteration variables to GPU thread indices
    s[B].bind(hpo, block_y)
    s[B].bind(wpo, block_x)
    s[B].bind(kpo, block_z)
    
    #tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    #txz, wi = s[B].split(wi, nparts=vthread)  # virtual thread split
    #ty, fi = s[B].split(hpi, nparts=num_thread)
    #tx, wi = s[B].split(wpi, nparts=num_thread)
    #tx, wi = s[B].split(kpi, nparts=num_thread)
    #s[B].reorder(bz, by, bx, tyz, txz, ty, fi, ni)
    #s[B].reorder(bz,by,bx,wi, fi,ty)
    #s[B].reorder(bz,by,bx,ty,wi, fi)
    
    #s[B].bind(tyz, thread_yz)
    #s[B].bind(txz, thread_xz)
    s[B].bind(hpi, thread_y)
    s[B].bind(wpi, thread_x)
    s[B].bind(kpi, thread_z)
    s[B].reorder(wpi,hpo,kpo,hpi,wpo,hpii,kpi)
    
    
    # Schedule BL local write
    s[BL].compute_at(s[B],kpi)
    s[B].reorder(kpi,hpii)
    kp, hp, wp_p4  = s[BL].op.axis
    #wpo,wpi=s[BL].split(wp,factor=2)
    #hpo,hpi=s[BL].split(hp,factor=2)
    
    #s[B].reorder(wpo,hpo,kp,wpi,hpi,p4)
    wp,p4 = s[BL].split(wp_p4, factor=4)
    s[BL].reorder(hp,wp,p4)
    whp = s[BL].fuse(wp,hp)
    rc, = s[BL].op.reduce_axis
    rco,rci=s[BL].split(rc,factor=4)
    #s[BL].reorder(rco,rci,whp, p4)
    s[BL].reorder(rco,whp,rci,p4)
    s[BL].vectorize(p4)  # vectorize memory load
    #s[BL].vectorize(rci)  # vectorize memory load
    #s[BL].unroll(p4)
    s[BL].unroll(whp)
    s[BL].unroll(rci)
    
    # Attach computation to iteration variables
    #s[AA].compute_at(s[BL], rco)
    #s[WW].compute_at(s[BL], rco)
    s[AL].compute_at(s[BL], rco)
    s[WL].compute_at(s[BL], rco)
    
    kp, hp, wp_p4 = s[AL].op.axis
    #ty, ci = s[AA].split(ci, nparts=num_thread)
    #tx, ni = s[AA].split(ni, nparts=num_thread)
    wpo, wpi = s[AL].split(wp_p4, factor=4)
    #s[AA].reorder(ty, tx, yi, xi, ci, ni)
    #s[AA].bind(ty, thread_y)
    #s[AA].bind(tx, thread_x)
    s[AL].vectorize(wpi)  # vectorize memory load
    s[AL].unroll(wpo)
    s[AL].unroll(hp)
    
    # Schedule for W's shared memory load
    kp, cp = s[WL].op.axis
    #ty, ci = s[WL].split(ci, nparts=num_thread)
    cpo, cpi = s[WL].split(cp, factor=4)
    #tx, fi = s[WW].split(fi, nparts=num_thread)
    #s[WW].reorder(ty, tx, yi, xi, ci, fi)
    #s[WW].bind(ty, thread_y)
    #s[WW].bind(tx, thread_x)
    s[WL].vectorize(cpi)  # vectorize memory load
    s[WL].unroll(kp)
    
    wpio,wpii = s[B].split(wp4, factor=4)
    s[B].vectorize(wpii)  # vectorize memory load
    s[B].unroll(wpio)  # vectorize memory load
    s[B].unroll(hpii)  # vectorize memory load
    
    
    #print(tvm.lower(s, [A, W, B], simple_mode=True))
    #exit(0)
    
    
    ###############################################################################
    # Generate CUDA Kernel
    # --------------------
    #
    # Finally we use TVM to generate and compile the CUDA kernel, and evaluate the
    # latency of convolution.
    #
    #print(tvm.lower(s, [A, W, B], simple_mode=True))


    return s, [A,W, B]

def test_rpc_module():
    # graph
    open_image=1
    ddtype='float32'
    if open_image == 1:
        ddtype = "climgfloatr32"
    #n_num=1024
    #n = tvm.runtime.convert(1024)
    #A = te.placeholder((n,n), name="A", dtype=ddtype)
    #B = te.compute(A.shape, lambda i,j: A(i,j) + 1.0, name="B")
    #B.dtype = "float32"
    #a_np = np.random.uniform(size=(n_num,n_num)).astype("float32")
    temp = utils.tempdir()

    ## Establish remote connection with target hardware
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    remote = tracker.request(key, priority=0, session_timeout=60)

    ## Compile the Graph for CPU target
    #s = te.create_schedule(B.op)
    #xo, xi = s[B].split(B.op.axis[0], factor=64)
    #s[B].parallel(xi)
    #s[B].pragma(xo, "parallel_launch_point")
    #s[B].pragma(xi, "parallel_barrier_when_finish")
    #f = tvm.build(s, [A, B], target, name="myadd_cpu")
    #path_dso_cpu = temp.relpath("cpu_lib.so")
    #f.export_library(path_dso_cpu, ndk.create_shared)

    # Execute the portable graph on cpu target
    #print("Run CPU test ...")
    #ctx = remote.cpu(0)
    #remote.upload(path_dso_cpu)
    #f2 = remote.load_module("cpu_lib.so")
    #a = tvm.nd.array(a_np, ctx, A.dtype)
    #b = tvm.nd.array(np.zeros((n_num,n_num), dtype="float32"), ctx)
    #time_f = f2.time_evaluator(f2.entry_name, ctx, number=10)
    #cost = time_f(a, b).mean
    #print("%g secs/op\n" % cost)
    #np.testing.assert_equal(b.asnumpy(), a.asnumpy() + 1)

    print("test opencl")
    # Compile the Graph for OpenCL target
    if test_opencl:
        N, H, W, CO, CI, KH, KW, strides, padding = 1, 64, 64, 512, 256, 1, 1, (1, 1), (0, 0)
        PACK4 = 4
        W_P = H
        H_P = H
        C_P = CI//PACK4
        K_P = CO//PACK4

        with tvm.target.Target("opencl"):
            s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
            lib = tvm.build(s, arg_bufs, target_host=target_host)
            f=lib
        print(f.imported_modules[0].get_source()) if len(f.imported_modules) > 0 else print("source not imported")
        path_dso_cl = temp.relpath("dev_lib_cl.so")
        filename="dev_lib_cl.so"
        f.export_library(path_dso_cl, ndk.create_shared)
        remote.upload(temp.relpath(filename))

        a_np = np.arange(C_P*H_P*W_P*PACK4).reshape(C_P*PACK4,H_P*W_P)
        w_np = np.arange(CI*CO).reshape(K_P*4,C_P*4)
        a_tvm=a_np.T.reshape(C_P*H_P*W_P,4)
        B1=a_tvm[0::C_P,:]
        for i in range(C_P-1):
            B1=np.vstack((B1,a_tvm[i+1::C_P,:]))
        a_tvm_np=B1.reshape(C_P,H_P,W_P*PACK4)*1.0
        w_tvm_np=w_np.T*1.0
        # calculate the right answer
        A=a_np.astype("float32")
        W_NP=w_np.astype("float32")
        C=W_NP.dot(A)
        C=C.reshape(K_P*PACK4,H_P*W_P).T.reshape(K_P*H_P*W_P,4)
        B1=C[0::K_P,:]
        for i in range(K_P-1):
            B1=np.vstack((B1,C[i+1::K_P,:]))
        c_np=B1.reshape(K_P,H_P*W_P*PACK4)
        #c_np = conv2d_nchw_python(a_np, w_np, strides, padding)
        target = tvm.target.Target("opencl")

        ctx = remote.context(str(target), 0)
        rlib = remote.load_module(filename)

        a_tvm = tvm.nd.array(a_tvm_np, ctx=ctx, dtype = arg_bufs[0].dtype)
        w_tvm = tvm.nd.array(w_tvm_np, ctx=ctx, dtype = arg_bufs[1].dtype)
        c_tvm = tvm.nd.empty((K_P, H_P,W_P*PACK4), ctx=ctx,dtype = arg_bufs[2].dtype)

        time_f = rlib.time_evaluator(rlib.entry_name, ctx, number=3)
        cost = time_f(a_tvm, w_tvm, c_tvm).mean
        print("Time cost of this operator: %f,%f GFLOPs" % (cost,1* H* W* 512* 256*2/cost/1000000000))
        c_tvm = c_tvm.asnumpy().reshape(K_P,H_P*W_P*PACK4)
        tvm.testing.assert_allclose(c_np, c_tvm, rtol=1e-2)
        print("answer check passed")


if __name__ == "__main__":
    test_rpc_module()
