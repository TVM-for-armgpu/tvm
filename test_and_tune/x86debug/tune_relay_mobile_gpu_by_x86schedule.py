import os
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib import utils, ndk
from tvm.contrib.utils import tempdir
import tvm.contrib.graph_runtime as runtime

######################################################################
# Load a test image
# -----------------
# A single cat dominates the examples!
def load_image():
    from PIL import Image
    import numpy as np

    image_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
    image_path = download_testdata(image_url, 'cat.png', module='data')
    resized_image = Image.open(image_path).resize((224, 224))
    image_data = np.asarray(resized_image).astype("float32")

    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)

    # Preprocess image as described here:
    # https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
    image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
    image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
    image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    return image_data


######################################################################
# Load pretrained TFLite model
# ----------------------------
# Load mobilenet V1 TFLite model provided by Google
def get_network():
    input_dtype = "float32"
    # auto-scheduler prefers NHWC layout
    layout = "NCHW"
    batch_size=1
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    mod, params = tvm.relay.testing.mobilenet.get_workload(
        batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
    )
    return mod, params, input_shape, input_dtype
    def extract(path):
        import tarfile
        if path.endswith("tgz") or path.endswith("gz"):
            dir_path = os.path.dirname(path)
            tar = tarfile.open(path)
            tar.extractall(path=dir_path)
            tar.close()
        else:
            raise RuntimeError('Could not decompress the file: ' + path)

    from tvm.contrib.download import download_testdata

    model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"

    # Download model tar file and extract it to get mobilenet_v1_1.0_224.tflite
    model_path = download_testdata(model_url, "mobilenet_v1_1.0_224.tgz", module=['tf', 'official'])
    model_dir = os.path.dirname(model_path)
    extract(model_path)

    # Now we can open mobilenet_v1_1.0_224.tflite
    tflite_model_file = os.path.join(model_dir, "mobilenet_v1_1.0_224.tflite")
    tflite_model_buf = open(tflite_model_file, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # TFLite input tensor name, shape and type
    input_tensor = "input"
    input_shape = (1, 224, 224, 3)
    input_dtype = "float32"

    # Parse TFLite model and convert it to a Relay module
    from tvm import relay
    mod, params = relay.frontend.from_tflite(tflite_model,
                                             shape_dict={input_tensor: input_shape},
                                             dtype_dict={input_tensor: input_dtype})
    return mod, params, input_shape, input_dtype


#################################################################
# Replace "armv8a-linux-gnueabihf" with the correct target of your board.
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
target = tvm.target.Target('llvm  -device=arm_cpu -mtriple=arm64-linux-android')

arch = "arm64"
target_host = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)
target = tvm.target.Target("opencl -device=mali")

# Also replace this with the device key in your tracker
device_key = "Adreno640"


#### TUNING OPTION ####
network = 'mobilenet_v1_1.0_224'
log_file = "%s.%s.log" % (device_key, network)
dtype = 'float32'

use_android=True
tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 15,
    'early_stopping': 800,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func='ndk' if use_android else 'default'),
        runner=autotvm.RPCRunner(
            device_key, host='0.0.0.0', port=9090,
            number=5,
            timeout=10,
        ),
    ),
}

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
#   If your model has depthwise convolution, you could consider setting
#   :code:`try_spatial_pack_depthwise` be :code:`True`, which perform better than default
#   optimization in general. For example, on ARM CPU A53 2.0GHz, we find it could boost 1.6x
#   performance of depthwise convolution on Mobilenet V1 model.

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    #if os.path.exists(tmp_log_file):
    #    os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    #os.remove(tmp_log_file)

# At commit baff99c83f9f691174434e7c78a4fee48b558547, ARM NHWC schedule is not high performance. So,
# we first switch to NCHW. Further, Relay build calls AlterOpLayout to optimize the data layout to
# NCHWc. If you want to use NHWC layout, please set use_nchw to False.
use_nchw = True

# TVM has many conv2d schedules for different platforms. As of commit
# baff99c83f9f691174434e7c78a4fee48b558547, we observed that x86 NCHWc schdules are faster than ARM
# NCHW or ARM NHWC schedule. If you want to use ARM NCHW spatial pack schedule, set this to false.
use_x86_schedules = True
if use_x86_schedules:
    # We must convert to NCHW first to use x86 schedules
    assert use_nchw
def remove_template(tasks, template_names):
    """ Removes the tasks that have a template name present in the template_names list.
    Parameters
    ----------
    tasks: list of tasks.
        The list of autotvm tasks.
    template_names : list of str.
        The list of template names that should be removed from tasks.
    Returns
    -------
    out_tasks: list of tasks.
        Filtered out tasks.
    """

    out_tasks = []
    for task in tasks:
        template_name = task.workload[0]
        if template_name not in template_names:
            out_tasks.append(task)
    return out_tasks


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.
def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, _ = get_network()

    if use_nchw:
        # Convert the layout to NCHW
        desired_layouts = {"nn.conv2d": ["NCHW","OIHW"]}
        seq = tvm.transform.Sequential([relay.transform.RemoveUnusedFunctions(),
                                        relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=2):
            mod = seq(mod)

    tasks = autotvm.task.extract_from_program(mod["main"], target=target, target_host=target_host,
                                              params=params,
                                              ops=(relay.op.get("nn.conv2d"),))


    if use_x86_schedules:
        tasks = remove_template(tasks, ["conv2d_nchw_spatial_pack.arm_cpu",
                                                     "depthwise_conv2d_nchw.arm_cpu"])

    # run tuning tasks
    print("Tuning...")
    #tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=2):
            graph, lib, params = relay.build_module.build(
                mod, target=target, target_host=target_host, params=params)

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
        remote = autotvm.measure.request_remote(device_key, '0.0.0.0', 9090,
                                                timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        ctx = remote.context(str(target), 0)
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.
if __name__ =='__main__':
    tune_and_evaluate(tuning_option)

