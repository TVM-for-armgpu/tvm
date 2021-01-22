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
 * \file codegen_opencl.cc
 */
#include "codegen_opencl.h"

#include <cmath>
#include <string>
#include <vector>

#include "../../runtime/opencl/opencl_module.h"
#include "../../runtime/thread_storage_scope.h"
#include "../build_common.h"
#include <tvm/tir/expr_simplify.h>

namespace tvm {
namespace codegen {

CodeGenOpenCL::CodeGenOpenCL() { restrict_keyword_ = "restrict"; }

void CodeGenOpenCL::InitFuncState(const PrimFunc& f) {
  CodeGenC::InitFuncState(f);
  for (Var arg : f->params) {
    if (arg.dtype().is_handle()) {
      int vcode = GetValueType(GetType(arg));
      if (kDLCLImgFloat == vcode) {
        alloc_storage_scope_[arg.get()] = "imager";
      } else if (kDLCLImgFloatW == vcode) {
        alloc_storage_scope_[arg.get()] = "imagew";
      } else {
        alloc_storage_scope_[arg.get()] = "global";
      }
    }
  }
}

void CodeGenOpenCL::PrintFuncPrefix() {
  stream << "__kernel void";
}
void CodeGenOpenCL::PreFunctionBody(const PrimFunc& f) {
  in_para_stm = false;
  stream << R"(
    const int global_id0 = get_global_id(0);
    const int global_id1 = get_global_id(1);
    const int global_id2 = get_global_id(2);

    const int group_id0 = get_group_id(0);
    const int group_id1 = get_group_id(1);
    const int group_id2 = get_group_id(2);

    const int local_id0 = get_local_id(0);
    const int local_id1 = get_local_id(1);
    const int local_id2 = get_local_id(2);

    )";
}
void CodeGenOpenCL::PrintGlobalSamplerDeclare() {
  stream << "__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | "
            "CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;\n";
}

std::string CodeGenOpenCL::Finish() {
  // inject extension enable pragma for fp16 and fp64
  if (enable_fp16_) {
    decl_stream << "#ifdef cl_khr_fp16\n"
                   "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
                   "#elif defined(cl_amd_fp16)\n"
                   "#pragma OPENCL EXTENSION cl_amd_fp16 : enable\n"
                   "#else\n"
                   "#error \"Half precision floating point not supported"
                   "by OpenCL implementation on your device.\" \n"
                   "#endif\n\n";
  }

  if (enable_fp64_) {
    decl_stream << "#ifdef cl_khr_fp64\n"
                   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
                   "#elif defined(cl_amd_fp64)\n"
                   "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
                   "#else\n"
                   "#error \"Double precision floating point not supported"
                   "by OpenCL implementation on your device.\" \n"
                   "#endif\n\n";
  }

  // Enable atomic_add used by get_valid_counts. Only needed for OpenCL < 1.1.
  if (enable_atomics_) {
    decl_stream << "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n"
                   "#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable\n\n";
  }
  return CodeGenC::Finish();
}

void CodeGenOpenCL::BindThreadIndex(const IterVar& iv) {
  ICHECK(!var_idmap_.count(iv->var.get()));
  runtime::ThreadScope ts = runtime::ThreadScope::Create(iv->thread_tag);
  std::ostringstream os;
  if (ts.rank == 1) {
    //os << "get_local_id(" << ts.dim_index << ")";
    os << "local_id" << ts.dim_index;
  } else {
    //os << "get_group_id(" << ts.dim_index << ")";
    os << "group_id" << ts.dim_index;
  }
  var_idmap_[iv->var.get()] = CastFromTo(os.str(), DataType::UInt(64), iv->var.dtype());
}

void CodeGenOpenCL::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "do not yet support vector types";
    os << "void*";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (in_para_stm && (t.is_climgfloat() || t.is_climgfloatw())) {
    if (t.lanes() != 4) {
      //LOG(WARNING) << "you are using " << t << "*" << t.lanes()
      //             << " as opencl image object type!!!!!!!";
      if (t.lanes() != 1) {
        LOG(FATAL) << "Cannot convert type " << t << "*" << t.lanes()
                   << " to OpenCL image object type, only " << t << "*"
                   << "4 is support ";
        return;
      }
    }
    os << "image2d_t";
    return;
  }
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        enable_fp16_ = true;
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        enable_fp64_ = true;
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    if (t.bits() == 8 && t.lanes() == 4) {
      // directly 4 8 bit int in integer.
      os << "int";
      return;
    }
    switch (t.bits()) {
      case 8:
        os << "char";
        break;
      case 16:
        os << "short";
        break;
      case 32:
        os << "int";
        break;
      case 64:
        os << "long";
        break;
      case 1:
        os << "int";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to OpenCL type";
}

// for string delimiter
static std::vector<std::string> split(std::string s, std::string delimiter) {
  size_t pos_start = 0, pos_end;
  std::string token;
  std::vector<std::string> res;

  for (pos_end = 0; pos_end < s.size(); ++pos_end) {
    if (delimiter.find(s[pos_end]) != std::string::npos) {
      if (pos_end - pos_start != 0) {
        res.emplace_back(s.substr(pos_start, pos_end - pos_start));
        pos_start = pos_end;
      }
    }
  }
  if (pos_start < s.size()) {
    res.push_back(s.substr(pos_start));
  }
  return res;
}

 void trimSpace(std::string &s)
 {
    size_t index = 0;
    if( !s.empty())
    {
        while( (index = s.find(' ',index)) != std::string::npos)
        {
            s.erase(index,1);
        }
    }
}

 bool CodeGenOpenCL::find_longst_common_str_or_add_key(const std::string& base,
                                                      std::string& new_base_index) {
  //for constant expr, we just skip it
  if (base.find_first_of("+*-/%") == std::string::npos) {
    return false;
  }
  
  std::string remove_space = base;
  trimSpace(remove_space);

  std::vector<std::string> vec_base = split(remove_space, "+*-/%");
  std::vector<std::string> vec_combine;
  for (const auto& v : vec_base) {
    if (vec_combine.empty()) {
      vec_combine.push_back(v);
    } else {
      vec_combine.push_back(vec_combine.back() + v);
    }
  }
  for (auto vec_comb_it = vec_combine.rbegin(); vec_comb_it != vec_combine.rend(); ++vec_comb_it) {
    for (const auto& kv : var_declare_map_) {
      size_t pos = vec_comb_it->find(kv.first);
      if (pos != std::string::npos) {
        if (vec_comb_it->size() - kv.first.size() >= 0 &&
            vec_comb_it->substr(0, pos).find_first_not_of("()") == std::string::npos) {
          new_base_index = kv.second + remove_space.substr(pos + kv.first.size());
          int left_bracket = std::count(new_base_index.begin(), new_base_index.end(), '(');
          int right_bracket = std::count(new_base_index.begin(), new_base_index.end(), ')');
          new_base_index.append(std::string(std::max(0, left_bracket - right_bracket), ')'));
          new_base_index.insert(0, std::string(std::max(0, right_bracket - left_bracket), '('));
          return true;
        }
      }
    }
  }
  std::hash<std::string> hash_str;
  var_declare_map_[remove_space] = "g_" + std::to_string(hash_str(base));
  new_base_index = var_declare_map_[remove_space];
  return false;
}

 void split_img_xy_axes(std::string index_str, std::string& img_x_axes, std::string& img_y_axes) {
  trimSpace(index_str);
  size_t pos = index_str.find("%59");
  ICHECK(pos != std::string::npos) << "this should not happend";
  int rb = 0, lb = 0;
  int m_pos = 0;
  for (int i = pos - 1; i >= 0; i--) {
    if (index_str[i] == ')') {
      rb++;
    } else if (index_str[i] == '(') {
      lb++;
    }
    if (lb == rb) {
      m_pos = i;
      break;
    }
  }
  img_y_axes = index_str.substr(m_pos, pos - m_pos);
  img_x_axes = index_str.substr(0, m_pos) + "0" + index_str.substr(pos + 3);
}

void CodeGenOpenCL::PrintVecAddr(const VarNode* buffer, DataType t, PrimExpr base,
                                 std::ostream& os) {  // NOLINT(*)
  std::ostringstream ossbase;

  do {
    if (t.is_climgfloat() || t.is_climgfloatw()) {
      std::string vid = GetVarID(buffer);
      if (var_buffer_map_.find(vid) == var_buffer_map_.end()) {
        break;
      }
      if (!HandleTypeMatch(buffer, t.element_of())) {
        //os << '(';
        //auto it = alloc_storage_scope_.find(buffer);
        //if (it != alloc_storage_scope_.end()) {
          //PrintStorageScope(it->second, os);
        //}
        //PrintType(t.element_of(), os);
        //os << "*)";
      }
      os << vid << ",";
      if (t.is_climgfloat()) {
        os << "sampler, ";
      }
      std::ostringstream osindex;
      //osindex << new_base_index;
      PrintExpr(base, osindex);
      //===
      std::string img_x_axes, img_y_axes;
      split_img_xy_axes(osindex.str(), img_x_axes, img_y_axes);
      img_x_axes += "/4";
      //4 == lanes
      img_x_axes = tvm::tir::exprSimp::DoSimplify(img_x_axes);

      std::string new_base_index_x = img_x_axes;
      std::string new_base_index_y = img_y_axes;
      find_longst_common_str_or_add_key(img_x_axes, new_base_index_x);
      find_longst_common_str_or_add_key(img_y_axes, new_base_index_y);

      os << "(int2)(" << new_base_index_x << "," << new_base_index_y << ")";
      //===
      /*
      ICHECK(var_buffer_map_.find(vid) != var_buffer_map_.end())
          << "var buffer shape is essential for opencl var:" << vid;
      ICHECK(var_buffer_map_[vid]->shape.size() > 1)
          << "var buffer shape of image memory must be at least 2 dimention";
      PrimExpr width = var_buffer_map_[vid]->shape[1];
      PrimExpr channel = IntImm(DataType::Int(32), 4);
      if (var_buffer_map_[vid]->shape.size() > 2) {
        width = var_buffer_map_[vid]->shape[2];
      }
      int Quotient = Downcast<IntImm>(width)->value / Downcast<IntImm>(channel)->value;
      if (Quotient == 0) {
        Quotient = 1;
      }*/
      //os << "(int2)(" << osindex.str() << "/" << channel << "%(" << Quotient << "),"
      //   << osindex.str() << "/(" << width << "))";
      
      return;
    }
  } while (0);
  PrintExpr(base, ossbase);
  std::string new_base_index = ossbase.str();
  find_longst_common_str_or_add_key(ossbase.str(), new_base_index);
  
  if (!HandleTypeMatch(buffer, t.element_of())) {
    os << '(';
    auto it = alloc_storage_scope_.find(buffer);
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    PrintType(t.element_of(), os);
    os << "*)";
  }
  os << GetVarID(buffer) << " + ";
  os << new_base_index;
  //PrintExpr(base, os);
}
std::string CodeGenOpenCL::GetVecLoad(DataType t, const VarNode* buffer, PrimExpr base) {
  std::ostringstream os;
  if (t.is_climgfloat()) {
    os << "read_imagef(";
  } else {
    os << "vload" << t.lanes() << "(0, ";
  }
  PrintVecAddr(buffer, t, base, os);
  os << ")";
  return os.str();
}

void CodeGenOpenCL::PrintVecStore(const VarNode* buffer, DataType t, PrimExpr base,
                                  const std::string& value) {
  this->PrintIndent();
  std::ostringstream os;
  
  std::string vid = GetVarID(buffer);
  DataType nt = t;
  if (var_buffer_map_.count(vid)) {
    nt = var_buffer_map_[vid]->dtype;
  }
  if (value.find("xyindex") != std::string::npos) {
    stream << need_declar_value_;
    need_declar_value_ = "";
    this->PrintIndent();
  }
  if (nt.is_climgfloatw() ||  t.is_climgfloatw()) {
    stream << "write_imagef(";
    // don't know why t's type was eliminated
    PrintVecAddr(buffer, nt, base, stream);
    stream << "," << value;
  } else {
    stream << "vstore" << t.lanes() << "(" << value << ", 0, ";
    PrintVecAddr(buffer, nt, base, stream);
  }

  stream << ");\n";
}

void CodeGenOpenCL::PrintStorageSync(const CallNode* op) {
  const std::string& sync = op->args[0].as<StringImmNode>()->value;
  if (sync == "warp") {
    this->PrintIndent();
    this->stream << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  } else if (sync == "shared") {
    this->PrintIndent();
    this->stream << "barrier(CLK_LOCAL_MEM_FENCE);\n";
  } else if (sync == "global") {
    LOG(FATAL) << "not supported";
  }
}

void CodeGenOpenCL::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  if (scope == "global") {
    os << "__global ";
  } else if (scope == "shared") {
    os << "__local ";
  } else if (scope == "imager") {
    os << "__read_only ";
  } else if (scope == "imagew") {
    os << "__write_only ";
  }
}

std::string CodeGenOpenCL::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  std::ostringstream os;
  if (target.lanes() == 1) {
    os << "((";
    this->PrintType(target, os);
    os << ")" << value << ")";
  } else {  // convert vector type
    os << "(";
    os << "convert_";
    this->PrintType(target, os);
    os << "(" << value << "))";
  }
  return os.str();
}

void CodeGenOpenCL::VisitExpr_(const CallNode* op, std::ostream& os) {
  if (op->op.same_as(builtin::address_of())) {
    // Overload tvm_address_of to add storage scope (e.g. __global).
    const LoadNode* load = op->args[0].as<LoadNode>();
    ICHECK(op->args.size() == 1 && load);
    os << "((";
    auto it = alloc_storage_scope_.find(load->buffer_var.get());
    if (it != alloc_storage_scope_.end()) {
      PrintStorageScope(it->second, os);
    }
    this->PrintType(load->dtype.element_of(), os);
    os << " *)" << this->GetVarID(load->buffer_var.get()) << " + ";
    this->PrintExpr(load->index, os);
    os << ')';
  } else if (op->op.same_as(builtin_call_extern_)) {
    auto func = Downcast<StringImm>(op->args[0]);
    // Enable atomics extension if used.
    if (func->value == "atomic_add") {
      enable_atomics_ = true;
    }
    CodeGenC::VisitExpr_(op, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenOpenCL::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  size_t vid_pos = v.find('[');
  do {
        if (op->lanes == 4 && vid_pos != std::string::npos) {
        std::string vid = v.substr(0, vid_pos);
          std::string index_str = v.substr(vid_pos + 1, v.size() - vid_pos - 1 - 1);
        std::replace(index_str.begin(), index_str.end(), '(', ' ');
        std::replace(index_str.begin(), index_str.end(), ')', ' ');
        if (index_str.find_first_not_of("0123456789 ") != std::string::npos) {
          os << "((";
          PrintType(op->dtype.with_lanes(1), os);
          os << "*)(" << vid << "))[" << index_str << "]";
          return;
        }
        os << "((";
        PrintType(op->dtype, os);
        int ind = std::stoi(index_str);
        int indv = ind / 4;
        int modv = ind % 4;
        char subscript[4] = {'x', 'y', 'z', 'w'};
        os << "*)(" << vid << "))[" << indv << "]." << subscript[modv];
        return;
      }
  } while (0);
  os << "((";
  PrintType(op->dtype, os);
  os << ")(";
  for (int i = 0; i < op->lanes; ++i) {
    if (i != 0) os << ", ";
    os << v;
  }
  os << "))";
}

void CodeGenOpenCL::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  if (std::isinf(op->value)) {
    if (op->value < 0) {
      os << "-";
    }
    os << "INFINITY";
  } else if (std::isnan(op->value)) {
    os << "NAN";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}
// Print a reference expression to a buffer.
std::string CodeGenOpenCL::GetBufferRef(DataType t, const VarNode* buffer, PrimExpr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  std::string scope;
  if (alloc_storage_scope_.count(buffer)) {
    scope = alloc_storage_scope_.at(buffer);
  }
  if (scope.compare(0,sizeof("image")-1, "image") != 0) {
    return CodeGenC::GetBufferRef(t, buffer, index);
  }
  bool is_vol = IsVolatile(buffer);
  if (t.lanes() == 1) {
    if (!HandleTypeMatch(buffer, t) || is_vol) {
      os << "((";
      if (is_vol) {
        os << "volatile ";
      }
      // Scope may not be part of type.
      if (!scope.empty() && IsScopePartOfType()) {
        PrintStorageScope(scope, os);
      }
      PrintType(t, os);
      // os << "*)" << vid << ')';
    } else {
      // os << vid;
    }
    // os << "[(";
    //const AddNode* ad = static_cast<const AddNode*>(index.get());

    // PrintExpr(index, os);
    // os << ")";
    if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
      os << " / " << (32 / t.bits());
    }
    // os << ']';
    //os << "read_imagef(" << vid << ",sampler,"
    os << "(int2)(";
    std::ostringstream indexexp_os;
    PrintExpr(index, indexexp_os);
    ICHECK(var_buffer_map_.find(vid) != var_buffer_map_.end())
        << "var buffer shape is essential for opencl var";

    ICHECK(var_buffer_map_[vid]->shape.size() > 1)
        << "var buffer shape of image memory must be at least 2 dimention";
    PrimExpr width = var_buffer_map_[vid]->shape[1];
    //how many elements in image object,default is CL_RGBA by 4
    PrimExpr channel = IntImm(DataType::Int(32), 4);
    if (var_buffer_map_[vid]->shape.size() > 2) {
      width = var_buffer_map_[vid]->shape[2] ;
    }
    int Quotient = Downcast<IntImm>(width)->value / Downcast<IntImm>(channel)->value;
    if (Quotient == 0) {
      Quotient = 1;
    }
    //===
    std::string img_x_axes, img_y_axes;
    split_img_xy_axes(indexexp_os.str(), img_x_axes, img_y_axes);
    img_x_axes += "/4";
    img_x_axes = tvm::tir::exprSimp::DoSimplify(img_x_axes);
    std::string new_base_index_x = img_x_axes;
    std::string new_base_index_y = img_y_axes;
    find_longst_common_str_or_add_key(img_x_axes, new_base_index_x);
    find_longst_common_str_or_add_key(img_y_axes, new_base_index_y);
    os << new_base_index_x << "," << new_base_index_y << ")";
    LOG(WARNING) << "answer will not correct!";
    //===
    //std::string xyindex = GetUniqueName("xyindex");
    //os << xyindex << "/" << channel << "%(" << Quotient << "),";
    //os << xyindex << "/(" << width << "))";
    //need_declar_value_ = "int "+ xyindex + "=" + indexexp_os.str() + ";\n";
  } else {
    // Buffer declared as vector type.
    // optimize for case where it is in register,
    if (HandleTypeMatch(buffer, t) && !is_vol) {
      // optimize for constant access
      if (auto* ptr = index.as<tir::IntImmNode>()) {
        int64_t offset = ptr->value;
        ICHECK_EQ(offset % t.lanes(), 0) << "Find unaligned vector load to a vector type";
        os << vid << '[' << (offset / t.lanes()) << ']';
        return os.str();
      }
    }
    os << "((";
    if (is_vol) {
      os << "volatile ";
    }
    if (!scope.empty() && IsScopePartOfType()) {
      PrintStorageScope(scope, os);
    }
    PrintType(t, os);
    os << "*)(";
    if (!HandleTypeMatch(buffer, t.element_of())) {
      os << '(';
      if (!scope.empty() && IsScopePartOfType()) {
        PrintStorageScope(scope, os);
      }
      PrintType(t.element_of(), os);
      os << "*)";
    }
    os << vid << " + (";
    PrintExpr(index, os);
    os << ")";
    if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
      os << " / " << (32 / t.bits());
    }
    os << "))[0]";
  }
  return os.str();
}

runtime::Module BuildOpenCL(IRModule mod, Target target) {
  using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenOpenCL cg;
  cg.Init(output_ssa);
  cg.PrintGlobalSamplerDeclare();
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenOpenCL: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenOpenCL: expect calling_conv equals CallingConv::kDeviceKernelLaunch";
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();
  if (const auto* f = Registry::Get("tvm_callback_opencl_postproc")) {
    code = (*f)(code).operator std::string();
  }
  return OpenCLModuleCreate(code, "cl", ExtractFuncInfo(mod), code);
}

TVM_REGISTER_GLOBAL("target.build.opencl").set_body_typed(BuildOpenCL);
}  // namespace codegen
}  // namespace tvm
