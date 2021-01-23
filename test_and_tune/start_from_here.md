## guideline of tVM-for-ARM-gpu


## build TVM
### 1. adjust data sctructure for dlpack
you need to add a change manully for the 3rd-parth lib.
**3rdparty/dlpack/include/dlpack/dlpack.h** 
 ```
/*!
 * \brief The type code options DLDataType.
 */
typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLBfloat = 4U,
  kDLCLImgFloatW = 126U,//new
  kDLCLImgFloat = 127U,//new
} DLDataTypeCode;
```

Please refer to the official document [click-and-go](https://tvm.apache.org/docs/install/from_source.html)

### Config Cmakefile
**attention**: you need to build for x86(your host PC) and arm(your device)

**for x86**
- turn off opencl, otherwise you will get runtime error.
- config llvm
- [config.cmake](./x86config.cmake)

**for arm**
- turn on opencl
- turn on cpp-rpc
- [config.cmake](./armconfig.cmake)

 [arm_runtime](./arm_runtime) may help you to onboard easily
## Setup the remote rpc environment
### option 1 NIC enabled
if you could alloc a Network interface controller for andriod and make the host and device connect together. 

### option 2 NIC not enabled
you have to forward and revert the port through ADB tools. So we could also have bi-tcp-link between host and device 

Suppose the ip address in host is '192.168.42.2', '192.168.42.129' in device. make sure you could visit 192.168.42.129:9000(rpc server running on device) from host, and you could visit 192.168.42.2:9090(rpc trakcer server will run on host) from device.

[Here](https://github.com/apache/tvm/blob/main/apps/android_rpc/README.md) is a instruction from official.

## Tunning RVM for ADRENO and MALI
[mad instruction support](./extern_op.py) TVM doesnot support MAD(a*b+c), so we provide a implementation here.

[test_rpc](android_rpc_testimage.py) is a test_scipt for rpc

[4-channel image/buffer](./tunerelaymobilegpu_nhcw4c.py) is a good example to warm up, which can achive the best performance on both GPU.

[1-channel buffer](tunerelaymobilegpu_vec_buffer.py) will use shared-memory for cross-thread-reduction to achive a competitive result below 4-channle-image template.

## results
|  memory-type   | lanes | conv1x1 shape |GFLOPS|
|  ------------  | ----  |  ------------ |----|
| buffer         | 1     | 40*40 256 512 | 120|
| buffer         | 4     |40*40 256 512 | 120|
| image         | 1     |40*40 256 512 | 125|
| image         | 4     |40*40 256 512 | 263|