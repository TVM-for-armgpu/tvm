/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file opencl_device_api.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>

#include "opencl_common.h"

namespace tvm {
namespace runtime {
namespace cl {

OpenCLThreadEntry* OpenCLWorkspace::GetThreadEntry() { return OpenCLThreadEntry::ThreadLocal(); }

OpenCLWorkspace* OpenCLWorkspace::Global() {
  static OpenCLWorkspace* inst = new OpenCLWorkspace();
  return inst;
}

void OpenCLWorkspace::SetDevice(TVMContext ctx) {
  GetThreadEntry()->context.device_id = ctx.device_id;
}

void OpenCLWorkspace::GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) {
  this->Init();
  size_t index = static_cast<size_t>(ctx.device_id);
  if (kind == kExist) {
    *rv = static_cast<int>(index < devices.size());
    return;
  }
  ICHECK_LT(index, devices.size()) << "Invalid device id " << index;
  switch (kind) {
    case kExist:
      break;
    case kMaxThreadsPerBlock: {
      size_t value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t),
                                  &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kWarpSize: {
      /* TODO: the warp size of OpenCL device is not always 1
               e.g. Intel Graphics has a sub group concept which contains 8 - 32 work items,
               corresponding to the number of SIMD entries the heardware configures.
               We need to figure out a way to query this information from the hardware.
      */
      *rv = 1;
      break;
    }
    case kMaxSharedMemoryPerBlock: {
      cl_ulong value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),
                                  &value, nullptr));
      *rv = static_cast<int64_t>(value);
      break;
    }
    case kComputeVersion:
      return;
    case kDeviceName: {
      char value[128] = {0};
      OPENCL_CALL(
          clGetDeviceInfo(devices[index], CL_DEVICE_NAME, sizeof(value) - 1, value, nullptr));
      *rv = std::string(value);
      break;
    }
    case kMaxClockRate: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint),
                                  &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMultiProcessorCount: {
      cl_uint value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                                  &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMaxThreadDimensions: {
      size_t dims[3];
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(dims), dims,
                                  nullptr));

      std::stringstream ss;  // use json string to return multiple int values;
      ss << "[" << dims[0] << ", " << dims[1] << ", " << dims[2] << "]";
      *rv = ss.str();
      break;
    }
    case kCL_DEVICE_IMAGE2D_MAX_WIDTH: {
      size_t value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t),
                                  &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kCL_DEVICE_IMAGE2D_MAX_HEIGHT: {
      size_t value;
      OPENCL_CALL(clGetDeviceInfo(devices[index], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t),
                                  &value, nullptr));
      *rv = static_cast<int32_t>(value);
      break;
    }
    case kMaxRegistersPerBlock:
      return;
    case kGcnArch:
      return;
    case kApiVersion:
      return;
    default:
      LOG(FATAL) << "not surpport this type kind=" << kind;

  }
}

void* OpenCLWorkspace::AllocDataSpace(TVMContext ctx, size_t size, size_t alignment,
                                      DLDataType type_hint) {
  this->Init();
  ICHECK(context != nullptr) << "No OpenCL device";
  cl_int err_code;
  cl_mem mptr=0;
  if (size == 0) {
    size++;
    LOG(WARNING) << "attention:size == 0, may cause the wrong ans";
  }
  mptr = clCreateBuffer(this->context, CL_MEM_READ_WRITE, size, nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}

void OpenCLWorkspace::get_image_t_size(TVMContext ctx, DataShape* dsize, size_t& height,
                                       size_t& width) {
#if USE_CL_RGBA
  int lans = dsize->dtype.lanes;
  lans = 4;
  ICHECK(lans == 1 || lans == 4) << "opencl image only surpport CL_RGBA and CL_R now";
  ICHECK(dsize->ndim > 0 && dsize->ndim <= 6 ) << "opencl image memory shape must be at least 2D";
  if (dsize->ndim == 6) {
    width = dsize->shape[4] * dsize->shape[3] * dsize->shape[2] * dsize->shape[1] *
            dsize->shape[5] / lans;
    height = dsize->shape[0];
  } else if (dsize->ndim == 5) {
    //for nhcw4 mace  --> h==w and w!= c/4 then c/4*w*4 as width, n*c as heith
    if ((dsize->shape[3] == dsize->shape[1] && dsize->shape[3] != dsize->shape[2])
    || (dsize->shape[2]*dsize->shape[3] * dsize->shape[4] / lans < 12345)
    ){
      width = dsize->shape[2]*dsize->shape[3] * dsize->shape[4] / lans;
      height = dsize->shape[1] * dsize->shape[0];
    } else {
      width = dsize->shape[3] * dsize->shape[4] / lans;
      height = dsize->shape[2] * dsize->shape[1] * dsize->shape[0];
    }
  } else if (dsize->ndim == 4) {
      width = dsize->shape[2] * dsize->shape[3] / lans;
      height = dsize->shape[1] * dsize->shape[0];
  } else if (dsize->ndim == 3){
    width = dsize->shape[1] * dsize->shape[2] / lans;
    height = dsize->shape[0];
  } else if (dsize->ndim == 2) {
    width = dsize->shape[1] / lans;
    height = dsize->shape[0];
  } else {
    width = dsize->shape[0] / lans;
    height = 1;
  }
  #else
    //HWCN
  height = dsize->shape[0];
  width = dsize->shape[1];
  if (dsize->ndim > 2) {
    width *= dsize->shape[2];
  }
  if (dsize->ndim > 3) {
    height *= dsize->shape[3];
  }
  TVMRetValue imgh, imgw;
  GetAttr(ctx, kCL_DEVICE_IMAGE2D_MAX_HEIGHT, &imgh);
  GetAttr(ctx, kCL_DEVICE_IMAGE2D_MAX_WIDTH, &imgw);
  int iw = imgw.operator int();
  int ih = imgh.operator int();
  if (width >= iw) {
    ICHECK(width % 2 == 0) << "width must be devide by 2, or we can't split into two part";
    width /= 2;
    height *= 2;
  }
  CHECK_LE(width, iw) << "image width is wider than the image object limit";
  CHECK_LE(height, ih) << "image height is higher than the image object limit";
  return;
  #endif
}

void* OpenCLWorkspace::AllocDataSpace(TVMContext ctx, DataShape* dsize, size_t alignment,
                                      DLDataType type_hint) {
  this->Init();
  ICHECK(context != nullptr) << "No OpenCL device";
  cl_int err_code;
#if USE_CL_RGBA
  cl_image_format fmt = {CL_RGBA, CL_HALF_FLOAT};
 #else
  cl_image_format fmt = {CL_R, CL_FLOAT};
#endif
  if (type_hint.lanes == 4) {
    fmt.image_channel_order = CL_RGBA;
  }
  if (type_hint.bits == 16) {
    fmt.image_channel_data_type = CL_HALF_FLOAT;
  }
  size_t height = 0, width = 0;
  get_image_t_size(ctx, dsize, height, width);

  std::ostringstream os;
  for (int i = 0; i < dsize->ndim; ++i) {
    os << dsize->shape[i] << ",";
  }

  cl_mem_flags mf = CL_MEM_READ_WRITE;
  TVMRetValue imgh, imgw;
  GetAttr(ctx, kCL_DEVICE_IMAGE2D_MAX_HEIGHT, &imgh);
  GetAttr(ctx, kCL_DEVICE_IMAGE2D_MAX_WIDTH, &imgw);
  int iw = imgw.operator int();
  int ih = imgh.operator int();
  CHECK_LE(width, iw) << "image width is wider than the image object limit " << os.str();
  CHECK_LE(height, ih) << "image height is higher than the image object limit " << os.str();
  cl_image_desc desc = {CL_MEM_OBJECT_IMAGE2D,
                        width,
                        height,
                        0,
                        0,  // depth, array size (unused)
                        0,
                        0,
                        0,
                        0,
                        0};

  cl_mem mptr = clCreateImage(this->context, mf, &fmt, &desc, NULL, &err_code);
  //LOG(WARNING) << "image sie x=" << width << " y=" << height << " shape " << os.str() << " " << mptr;
  OPENCL_CHECK_ERROR(err_code);
  return mptr;
}
void OpenCLWorkspace::FreeDataSpace(TVMContext ctx, void* ptr) {
  // We have to make sure that the memory object is not in the command queue
  // for some OpenCL platforms.
  OPENCL_CALL(clFinish(this->GetQueue(ctx)));

  cl_mem mptr = static_cast<cl_mem>(ptr);
  OPENCL_CALL(clReleaseMemObject(mptr));
}

void OpenCLWorkspace::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                     size_t to_offset, size_t size, TVMContext ctx_from,
                                     TVMContext ctx_to, DLDataType type_hint,
                                     TVMStreamHandle stream) {
  this->Init();
  ICHECK(type_hint.code < kDLCLImgFloatW) << "opencl image datatype should not call this function<CopyDataFromTo>";
  ICHECK(stream == nullptr);
  if (IsOpenCLDevice(ctx_from) && IsOpenCLDevice(ctx_to)) {
    OPENCL_CALL(clEnqueueCopyBuffer(this->GetQueue(ctx_to),
                                    static_cast<cl_mem>((void*)from),  // NOLINT(*)
                                    static_cast<cl_mem>(to), from_offset, to_offset, size, 0,
                                    nullptr, nullptr));
  } else if (IsOpenCLDevice(ctx_from) && ctx_to.device_type == kDLCPU) {
    OPENCL_CALL(clEnqueueReadBuffer(this->GetQueue(ctx_from),
                                    static_cast<cl_mem>((void*)from),  // NOLINT(*)
                                    CL_FALSE, from_offset, size, static_cast<char*>(to) + to_offset,
                                    0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(ctx_from)));
  } else if (ctx_from.device_type == kDLCPU && IsOpenCLDevice(ctx_to)) {
    OPENCL_CALL(clEnqueueWriteBuffer(
        this->GetQueue(ctx_to), static_cast<cl_mem>(to), CL_FALSE, to_offset, size,
        static_cast<const char*>(from) + from_offset, 0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(ctx_to)));
  } else {
    LOG(FATAL) << "Expect copy from/to OpenCL or between OpenCL";
  }
}

void OpenCLWorkspace::CopyDataFromTo(const void* from, size_t from_offset, void* to,
                                     size_t to_offset, DataShape* dsize, TVMContext ctx_from,
                                     TVMContext ctx_to, DLDataType type_hint,
                                     TVMStreamHandle stream) {
  this->Init();
  ICHECK(dsize->ndim > 0 && dsize->ndim <= 6 ) << "opencl image memory shape must be at least 2D";
  size_t height = 0, width = 0;
  get_image_t_size(ctx_to, dsize, height, width);

  size_t origin[3] = {
      0,
      0,
      0,
  };
  //CHECK_LE(width, CL_DEVICE_IMAGE2D_MAX_WIDTH) << "image width is wider than the limit";
  //CHECK_LE(height, CL_DEVICE_IMAGE2D_MAX_HEIGHT) << "image height is higher than the limit";
  size_t region[3] = {width, height, 1};
  ICHECK(stream == nullptr);
  if (IsOpenCLDevice(ctx_from) && IsOpenCLDevice(ctx_to)) {
    OPENCL_CALL(clEnqueueCopyImage(this->GetQueue(ctx_to),
                                    static_cast<cl_mem>((void*)from),  // NOLINT(*)
                                   static_cast<cl_mem>(to), origin, origin, region, 0,
                                    nullptr, nullptr));
  } else if (IsOpenCLDevice(ctx_from) && ctx_to.device_type == kDLCPU) {
    /* Copy the input data to the input image */

    OPENCL_CALL(clEnqueueReadImage(this->GetQueue(ctx_from),
                                   static_cast<cl_mem>((void*)from),  // NOLINT(*)
                                   CL_FALSE, origin, region, 0, 0,
                                   static_cast<char*>(to) + to_offset, 0, nullptr, nullptr));
    OPENCL_CALL(clFinish(this->GetQueue(ctx_from)));
  } else if (ctx_from.device_type == kDLCPU && IsOpenCLDevice(ctx_to)) {
    if (type_hint.code == kDLCLImgFloatW) {
      return;
    }
    /* Copy the input data to the input image */
    OPENCL_CALL(clEnqueueWriteImage(
        this->GetQueue(ctx_to), static_cast<cl_mem>(to), CL_FALSE, origin, region, 0, 0,
        static_cast<const char*>(from) + from_offset, 0, nullptr, nullptr));

    OPENCL_CALL(clFinish(this->GetQueue(ctx_to)));
  } else {
    LOG(FATAL) << "Expect copy from/to OpenCL or between OpenCL";
  }
}

void OpenCLWorkspace::GetTc(TVMContext ctx_from, void* data_shape) {
  ICHECK(data_shape != nullptr) << "time array is null";
  *(double*)data_shape = tc_duration_s_;
  tc_duration_s_=0;
}

void OpenCLWorkspace::StreamSync(TVMContext ctx, TVMStreamHandle stream) {
  ICHECK(stream == nullptr);
  OPENCL_CALL(clFinish(this->GetQueue(ctx)));
}

void* OpenCLWorkspace::AllocWorkspace(TVMContext ctx, DataShape* size, DLDataType type_hint) {
  return GetThreadEntry()->pool.AllocWorkspace(ctx, size);
}

void* OpenCLWorkspace::AllocWorkspace(TVMContext ctx, size_t size, DLDataType type_hint) {
  return GetThreadEntry()->pool.AllocWorkspace(ctx, size);
}

void OpenCLWorkspace::FreeWorkspace(TVMContext ctx, void* data) {
  GetThreadEntry()->pool.FreeWorkspace(ctx, data);
}

typedef dmlc::ThreadLocalStore<OpenCLThreadEntry> OpenCLThreadStore;

OpenCLThreadEntry* OpenCLThreadEntry::ThreadLocal() { return OpenCLThreadStore::Get(); }

std::string GetPlatformInfo(cl_platform_id pid, cl_platform_info param_name) {
  size_t ret_size;
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::string GetDeviceInfo(cl_device_id pid, cl_device_info param_name) {
  size_t ret_size;
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, 0, nullptr, &ret_size));
  std::string ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetDeviceInfo(pid, param_name, ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_platform_id> GetPlatformIDs() {
  cl_uint ret_size;
  cl_int code = clGetPlatformIDs(0, nullptr, &ret_size);
  std::vector<cl_platform_id> ret;
  if (code != CL_SUCCESS) return ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetPlatformIDs(ret_size, &ret[0], nullptr));
  return ret;
}

std::vector<cl_device_id> GetDeviceIDs(cl_platform_id pid, std::string device_type) {
  cl_device_type dtype = CL_DEVICE_TYPE_ALL;
  if (device_type == "cpu") dtype = CL_DEVICE_TYPE_CPU;
  if (device_type == "gpu") dtype = CL_DEVICE_TYPE_GPU;
  if (device_type == "accelerator") dtype = CL_DEVICE_TYPE_ACCELERATOR;
  cl_uint ret_size;
  cl_int code = clGetDeviceIDs(pid, dtype, 0, nullptr, &ret_size);
  std::vector<cl_device_id> ret;
  if (code != CL_SUCCESS) return ret;
  ret.resize(ret_size);
  OPENCL_CALL(clGetDeviceIDs(pid, dtype, ret_size, &ret[0], nullptr));
  return ret;
}

bool MatchPlatformInfo(cl_platform_id pid, cl_platform_info param_name, std::string value) {
  if (value.length() == 0) return true;
  std::string param_value = GetPlatformInfo(pid, param_name);
  return param_value.find(value) != std::string::npos;
}

void OpenCLWorkspace::Init(const std::string& type_key, const std::string& device_type,
                           const std::string& platform_name) {
  if (initialized_) return;
  std::lock_guard<std::mutex> lock(this->mu);
  if (initialized_) return;
  if (context != nullptr) return;
  this->type_key = type_key;
  // matched platforms
  std::vector<cl_platform_id> platform_ids = cl::GetPlatformIDs();
  if (platform_ids.size() == 0) {
    LOG(WARNING) << "No OpenCL platform matched given existing options ...";
    return;
  }
  this->platform_id = nullptr;
  for (auto platform_id : platform_ids) {
    if (!MatchPlatformInfo(platform_id, CL_PLATFORM_NAME, platform_name)) {
      continue;
    }
    std::vector<cl_device_id> devices_matched = cl::GetDeviceIDs(platform_id, device_type);
    if ((devices_matched.size() == 0) && (device_type == "gpu")) {
      LOG(WARNING) << "Using CPU OpenCL device";
      devices_matched = cl::GetDeviceIDs(platform_id, "cpu");
    }
    if (devices_matched.size() > 0) {
      this->platform_id = platform_id;
      this->platform_name = cl::GetPlatformInfo(platform_id, CL_PLATFORM_NAME);
      this->device_type = device_type;
      this->devices = devices_matched;
      break;
    }
  }
  if (this->platform_id == nullptr) {
    LOG(WARNING) << "No OpenCL device";
    return;
  }
  cl_int err_code;
  this->context = clCreateContext(nullptr, this->devices.size(), &(this->devices[0]), nullptr,
                                  nullptr, &err_code);
  OPENCL_CHECK_ERROR(err_code);
  ICHECK_EQ(this->queues.size(), 0U);
  for (size_t i = 0; i < this->devices.size(); ++i) {
    cl_device_id did = this->devices[i];
    this->queues.push_back(clCreateCommandQueue(this->context, did, CL_QUEUE_PROFILING_ENABLE, &err_code));
    OPENCL_CHECK_ERROR(err_code);
  }
  initialized_ = true;
}

TVM_REGISTER_GLOBAL("device_api.opencl").set_body([](TVMArgs args, TVMRetValue* rv) {
  DeviceAPI* ptr = OpenCLWorkspace::Global();
  *rv = static_cast<void*>(ptr);
});

}  // namespace cl
}  // namespace runtime
}  // namespace tvm
