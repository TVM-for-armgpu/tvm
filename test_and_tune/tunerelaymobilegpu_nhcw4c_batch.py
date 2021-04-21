import os
import time

import numpy as np

import tvm
from tvm import te
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime
from tvm import te, topi, testing
from tvm.topi.testing import conv2d_nchw_python
import extern_op

import logging
import sys
#logging.getLogger("autotvm").setLevel(logging.DEBUG)
#logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

TRACKER_PORT=9090
@autotvm.template("tutorial/conv2d_no_batching")
def conv2d_no_batching(N, H, W, CO, CI, KH, KW, stride, padding):
    assert N == 1, "Only consider batch_size = 1 in this template"

    cfg = autotvm.get_config()

    cfg.add_flop(2.0*np.prod([N, H, W, CO, CI, KH, KW]))
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
    W_P = in_size
    H_P = in_size
    C_P = in_channel//PACK4
    K_P=out_channel//PACK4

    if "Mali" in device_key:
        cfg.define_knob("idtype",[0,1])
        cfg.define_knob("kdtype",[0,1])
        typedict = {0:"float32",1:"climgfloatr32",2:"climgfloatw32"}
        A = te.placeholder((C_P,H_P,W_P*PACK4), dtype=typedict[cfg["idtype"].val],name="A")
        W = te.placeholder((in_channel,out_channel), dtype=typedict[cfg["kdtype"].val], name="W")
    else:
        A = te.placeholder((C_P,H_P,W_P*PACK4), dtype=ddtype,name="A1")
        W = te.placeholder((in_channel,out_channel), dtype=ddtype, name="W1")
    out_size = in_size
    # Pad input
    Apad = A
    # Create reduction variables
    rc = te.reduce_axis((0, in_channel), name="rc1")
    # Compute the convolution
    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    B = te.compute(
        (K_P, H_P,W_P*PACK4),
        lambda ff,yy,xx_p4: extern_op.mysum(
            extern_op.mymul(Apad[idxdiv(rc,4),yy,idxmod(rc,4)+idxdiv(xx_p4,4)*4] , W[rc,ff*PACK4+idxmod(xx_p4,4)]), axis=[rc]
        ),
        name="B1",
    )
    if open_image ==1:
        B.dtype="climgfloatw32"
    else:
        B.dtype="float32"


    s = te.create_schedule(B.op)
    # Split the workloads
    kp, hp, wp_p4  = s[B].op.axis
    rc, = s[B].op.reduce_axis


    cfg.define_split("tile_rc", rc,num_outputs=3,filter=lambda x: x.size[-1]==4)

    AL = s.cache_read(Apad, "local", [B])
    WL = s.cache_read(W, "local", [B])
    BL = s.cache_write(B, "local")

    # tile consts
    tile = 1
    num_thread = 4
    block_factor = tile * num_thread
    step = 4
    vthread = 2

    # Split the workloads
    kp, hp, wp_p4  = s[B].op.axis

    n, f, y, x = 1,kp,hp,wp_p4

    cfg.define_split("tile_f", f, num_outputs=2)
    cfg.define_split("tile_y", y, num_outputs=3,filter=lambda x: x.size[-1] %2==0)
    cfg.define_split("tile_x", x, num_outputs=3,filter=lambda x: x.size[-1] >=4 and x.size[-1]%4==0)

    bf, kpi = cfg["tile_f"].apply(s, B, f)
    by, vy, hpii = cfg["tile_y"].apply(s, B, y)
    bx, vx, wp4 = cfg["tile_x"].apply(s, B, x)

    s[B].bind(bf, te.thread_axis("blockIdx.z"))
    s[B].bind(by, te.thread_axis("blockIdx.y"))
    s[B].bind(bx, te.thread_axis("blockIdx.x"))

    s[B].bind(kpi, te.thread_axis("threadIdx.z"))
    s[B].bind(vy, te.thread_axis("threadIdx.y"))
    s[B].bind(vx, te.thread_axis("threadIdx.x"))
    #kernel_scope = n  # this is the scope to attach global config inside this kernel



    s[B].reorder(bf, by, bx, vy, vx, hpii,kpi)


    # Schedule BL local write
    s[BL].compute_at(s[B],kpi)
    s[B].reorder(kpi,hpii)
    kp, hp, wp_p4  = s[BL].op.axis

    wp,p4 = s[BL].split(wp_p4, factor=4)
    s[BL].reorder(hp,wp,p4)
    whp = s[BL].fuse(wp,hp)
    rc, = s[BL].op.reduce_axis
    rco,rcm,rci = cfg["tile_rc"].apply(s, BL, rc)
    s[BL].reorder(rco,rcm,rci,whp, p4)
    s[BL].vectorize(p4)  # vectorize memory load
    s[BL].unroll(whp)
    #s[BL].unroll(p4)
    s[BL].unroll(rci)
    s[BL].unroll(rcm)


    s[AL].compute_at(s[BL], rco)
    s[WL].compute_at(s[BL], rco)

    kp, hp, wp_p4 = s[AL].op.axis
    wpo, wpi = s[AL].split(wp_p4, factor=4)

    s[AL].vectorize(wpi)  # vectorize memory load
    s[AL].unroll(wpo)
    s[AL].unroll(hp)

    # Schedule for W's shared memory load
    kp, cp = s[WL].op.axis
    _, cpi = s[WL].split(cp, factor=4)
    s[WL].vectorize(cpi)  # vectorize memory load
    s[WL].unroll(kp)

    wpio,wpii = s[B].split(wp4, factor=4)
    s[B].vectorize(wpii)  # vectorize memory load
    s[B].unroll(wpio)  # vectorize memory load
    s[B].unroll(hpii)  # vectorize memory load
    #print(tvm.lower(s, [A,W, B], simple_mode=True))
    return s, [A,W, B]

#### DEVICE CONFIG ####

target = tvm.target.Target("opencl")
#target = tvm.target.Target("opencl -device=mali")

arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch


# Set this to True if you use android phone
use_android = True



####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default values provided here work well.
#   If you have enough time budget, you can set :code:`n_trial`, :code:`early_stopping` larger,
#   which makes the tuning run longer.
#   If your device runs very slow or your conv2d operators have many GFLOPs, considering to
#   set timeout larger.
#

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.
def in_check_point(shape:str) -> bool:
    #return False
    import json
    if isinstance(shape, tuple):
        shape = list(shape)

    if len(shape) == 11:
        shape.pop(1)
    #shape.pop(-1)
    for ind, sp in enumerate(shape):
        if isinstance(sp, tuple):
            shape[ind] = list(sp)
    if not os.path.exists(log_file):
        return False
    with open(log_file) as fp:
        for sp_hist in fp:
            sp_hist_json = json.loads(sp_hist)
            if sp_hist_json["input"][2] == shape:
                return True
    return False
# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    use_transfer_learning=False
    # create tmp log file
    tmp_log_file = 'tmp/' + log_filename + ".tmp"
    #if os.path.exists(tmp_log_file):
    #    os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        if in_check_point(tsk.args):
            print(f"skip finished {i}th task",tsk.args)
            continue
        print(f"start tuning {i}th task",tsk.args)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                print("load previous results")
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

        # pick best records to a cache file
        autotvm.record.pick_best(tmp_log_file, log_filename)
        #os.remove(tmp_log_file)
        time.sleep(60)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...",os.getpid())
    import conv_shape_tunning
    tasks=[]
    for shape in conv_shape_tunning.conv_shapes_1x1_d:
        N, _, H, W, CI, CO, KH, KW, strides, padding, _ = shape
        H=(1+H)//2*2
        W=(1+W)//2*2
        task = autotvm.task.create(
            "tutorial/conv2d_no_batching", args=(N, H, W, CO, CI, KH, KW, strides, padding), target=target,target_host=target_host
        )
        tasks.append(task)

    # run tuning tasks
    print("Tuning...")
    tasks.reverse()
    #tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file) as dispatch_context:
        best_config = dispatch_context.query(tasks[0].target, tasks[0].workload)
        print("\nBest config:")
        print(best_config)
        print("Compile...")
        with tvm.target.Target("opencl"):
            s, arg_bufs = conv2d_no_batching(N, H, W, CO, CI, KH, KW, strides, padding)
            lib = tvm.build(s, arg_bufs, target_host=target_host)
            func=lib
            print(func.imported_modules[0].get_source()) if len(func.imported_modules) > 0 else print("source not imported")
        # export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk

            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, "0.0.0.0", TRACKER_PORT, timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        a_s=tuple(int(i) for i in arg_bufs[0].shape)
        w_s=tuple(int(i) for i in arg_bufs[1].shape)
        #a_np = np.random.uniform(size=a_s).astype(np.float32)
        #w_np = np.random.uniform(size=w_s).astype(np.float32)
        # Algorithm
        PACK4 = 4
        W_P = W
        H_P = H
        C_P = CI//PACK4
        K_P=CO//PACK4

        a_np = np.arange(C_P*H_P*W_P*PACK4).reshape(C_P*PACK4,H_P*W_P)
        w_np = np.arange(CI*CO).reshape(K_P*4,C_P*4)
        ######## transform to the nchw4c layout
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
        ####numpy calculate the answer over

        #c_np = conv2d_nchw_python(a_np, w_np, strides, padding)
        ctx = remote.context(str(target), 0)
        a_tvm = tvm.nd.array(a_tvm_np, ctx=ctx, dtype=arg_bufs[0].dtype)
        w_tvm = tvm.nd.array(w_tvm_np, ctx=ctx, dtype = arg_bufs[1].dtype)
        c_tvm = tvm.nd.empty(arg_bufs[2].shape, ctx=ctx,dtype = arg_bufs[2].dtype)
        time_f = rlib.time_evaluator(rlib.entry_name, ctx, number=3)
        cost = time_f(a_tvm, w_tvm, c_tvm).mean
        print("Time cost of this operator: %f, %f" % (cost,W*H*CI*CO*2/cost/1e9))

        c_tvm_o = c_tvm.asnumpy().reshape(K_P,H_P*W_P*PACK4)
        tvm.testing.assert_allclose(c_np, c_tvm_o, rtol=1e-2)

        #ctx = remote.context(str(target), 0)
        #module = runtime.GraphModule(rlib["default"](ctx))
        #data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        #module.set_input("data", data_tvm)

        # evaluate
        print("answer check passed. Evaluate inference time cost... ")
        ftimer = rlib.time_evaluator(rlib.entry_name, ctx, number=10, repeat=30)
        a=ftimer(a_tvm, w_tvm, c_tvm).results
        prof_res = np.array(a) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms) %f GFLOPS"
            % (np.mean(prof_res), np.std(prof_res),W*H*CI*CO*2/np.mean(prof_res)/1e6)
        )


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.
if __name__ == '__main__':
    #### TUNING OPTION ####
    # Also replace this with the device key in your tracker
    device_key = "Adreno640"
    device_key = sys.argv[1]
    assert device_key in ["MaliG72", "MaliG76", "Adreno640", "Adreno630"]
    network = "topi_nchw4c"
    log_file = "%s.%s.log" % (device_key, network)

    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 1200,
        "use_transfer_learning":True,
        "early_stopping": 850,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
            runner=autotvm.RPCRunner(
                device_key,
                host="0.0.0.0",
                port=TRACKER_PORT,
                number=10,
                timeout=5,
            ),
        ),
    }
    tune_and_evaluate(tuning_option)
