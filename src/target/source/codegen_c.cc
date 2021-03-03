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
 * \file codegen_c.cc
 */
#include "codegen_c.h"

#include <cctype>
#include <iomanip>

#include "../../arith/pattern_match.h"
#include <tvm/tir/expr_simplify.h>
#include <regex>

namespace tvm {
namespace codegen {

using namespace tir;

void CodeGenC::Init(bool output_ssa) { print_ssa_form_ = output_ssa; }

void CodeGenC::InitFuncState(const PrimFunc& f) {
  for (auto& kv : f->clgen_buffer_map) {
    var_buffer_map_.Set(kv.first->name_hint, kv.second);
  }
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
}

void CodeGenC::ReserveKeywordsAsUnique() {
  // skip the first underscore, so SSA variable starts from _1
  GetUniqueName("_");
  GetUniqueName("extern");
  GetUniqueName("void");
  GetUniqueName("int");
  GetUniqueName("float");
  GetUniqueName("double");
  GetUniqueName("char");
  GetUniqueName("unsigned");
  GetUniqueName("short");
  GetUniqueName("long");
  GetUniqueName("if");
  GetUniqueName("else");
  GetUniqueName("switch");
  GetUniqueName("case");
  GetUniqueName("default");
  GetUniqueName("for");
  GetUniqueName("do");
  GetUniqueName("while");
  GetUniqueName("goto");
  GetUniqueName("register");
  GetUniqueName("continue");
  GetUniqueName("break");
  GetUniqueName("typedef");
  GetUniqueName("struct");
  GetUniqueName("enum");
  GetUniqueName("union");
  GetUniqueName("return");
}

int GetValueType(const Type& type) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return ptr->dtype.code();
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    return GetValueType(ptr->element_type);
  } else if (IsVoidType(type)) {
    return NULL;
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
    return 0;
  }
}

void CodeGenC::AddFunction(const PrimFunc& f) {
  // clear previous generated state.
  this->InitFuncState(f);
  // reserve keywords
  ReserveKeywordsAsUnique();

  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.defined())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";
  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);

  this->PrintFuncPrefix();
  this->stream << " " << static_cast<std::string>(global_symbol.value()) << "(";

  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());
    if (i != 0) stream << ", ";
    if (v.dtype().is_handle()) {
      auto it = alloc_storage_scope_.find(v.get());
      if (it != alloc_storage_scope_.end()) {
        PrintStorageScope(it->second, stream);
      }
      
      PrintType(GetType(v), stream);
      // Register handle data type
      // TODO(tvm-team): consider simply keep type info in the
      // type annotation(via a normalizing rewriting).
      if (auto* ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto* prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
      int vt_code = GetValueType(GetType(v));
      if ((kDLCLImgFloat != vt_code && kDLCLImgFloatW != vt_code) && no_alias && restrict_keyword_.length() != 0) {
        stream << ' ' << restrict_keyword_;
      }
    } else {
      PrintType(GetType(v), stream);
    }
    stream << ' ' << vid;
  }
  stream << ") {\n";

    //============hard kernel for winograd transform
  bool find_U = false, find_V = false, find_M = false;
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    auto getVarName = [](const tir::VarNode* v) { return v->name_hint; };
    if ("mali_conv2d_nchw_winograd_U" == getVarName(v.get())) {
      find_U = true;
    } else if ("mali_conv2d_nchw_winograd_V" == getVarName(v.get())) {
      find_V = true;
    } else if ("mali_conv2d_nchw_winograd_M" == getVarName(v.get())) {
      find_M = true;
    }
  }
  // only U,transorm filter
  if (find_U == true && (find_V | find_M) == false)
  {
    // NCHW1i4c
    this->stream << R"(

 #define filter_mat mali_conv2d_nchw_winograd_U 
      const int nk_pack4 = get_image_dim(filter).x/3/3 ;
      const int nc = (get_image_dim(filter).y);

       const int k_pack4 = get_global_id(0);
  const int c = get_global_id(1);
    //if (k_pack4==0 && c==0)
     //printf("mali_conv2d_nchw_winograd_U %d,%d",get_image_dim(mali_conv2d_nchw_winograd_U).x,get_image_dim(mali_conv2d_nchw_winograd_U).y);
     //return;
  if (k_pack4 >= nk_pack4 || c >= nc) return;

  float4 g00 = read_imagef(filter, sampler, (int2)(0 + k_pack4 * 9, c));
  float4 g01 = read_imagef(filter, sampler, (int2)(1 + k_pack4 * 9, c));
  float4 g02 = read_imagef(filter, sampler, (int2)(2 + k_pack4 * 9, c));
  float4 g10 = read_imagef(filter, sampler, (int2)(3 + k_pack4 * 9, c));
  float4 g11 = read_imagef(filter, sampler, (int2)(4 + k_pack4 * 9, c));
  float4 g12 = read_imagef(filter, sampler, (int2)(5 + k_pack4 * 9, c));
  float4 g20 = read_imagef(filter, sampler, (int2)(6 + k_pack4 * 9, c));
  float4 g21 = read_imagef(filter, sampler, (int2)(7 + k_pack4 * 9, c));
  float4 g22 = read_imagef(filter, sampler, (int2)(8 + k_pack4 * 9, c));

  float4 z00 = g00;
  float4 z01 = (g00 + g01 + g02) * 0.5f;
  float4 z02 = (g00 - g01 + g02) * 0.5f;
  float4 z03 = g02;
  float4 z10 = (g00 + g10 + g20) * 0.5f;
  float4 z11 = (g00 + g01 + g02 + g10 + g11 + g12 + g20 + g21 + g22) * 0.25f;
  float4 z12 = (g00 - g01 + g02 + g10 - g11 + g12 + g20 - g21 + g22) * 0.25f;
  float4 z13 = (g02 + g12 + g22) * 0.5f;
  float4 z20 = (g00 - g10 + g20) * 0.5f;
  float4 z21 = (g00 + g01 + g02 - g10 - g11 - g12 + g20 + g21 + g22) * 0.25f;
  float4 z22 = (g00 - g01 + g02 - g10 + g11 - g12 + g20 - g21 + g22) * 0.25f;
  float4 z23 = (g02 - g12 + g22) * 0.5f;
  float4 z30 = g20;
  float4 z31 = (g20 + g21 + g22) * 0.5f;
  float4 z32 = (g20 - g21 + g22) * 0.5f;
  float4 z33 = g22;

  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x0, c), z00);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x1, c), z01);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x2, c), z02);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x3, c), z03);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x4, c), z10);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x5, c), z11);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x6, c), z12);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x7, c), z13);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x8, c), z20);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0x9, c), z21);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xa, c), z22);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xb, c), z23);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xc, c), z30);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xd, c), z31);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xe, c), z32);
  write_imagef(filter_mat, (int2)(k_pack4 * 16 + 0xf, c), z33);
}

    )";
  } 
  else if (find_V == true && (find_U | find_M) == false) {
    // only V,transorm data
    // supporse w==h,otherwise it's not correct
    this->stream << R"(
        
    #define input data
    #define input_mat mali_conv2d_nchw_winograd_V 
    const int nc_pack4 = (get_image_dim(data).y/get_image_dim(data).x);
    const int nh = get_image_dim(data).x;
    const int nh_pack_wino = (nh+2-1)/2;
    const int nw_pack_wino = nh_pack_wino;


    const int w_pack_wino = get_global_id(0);
  const int h_pack_wino = get_global_id(1);
  const int c_pack4 = get_global_id(2);
//if (w_pack_wino==0&&h_pack_wino==0&&c_pack4==0)
//printf("mali_conv2d_nchw_winograd_V %d,%d",get_image_dim(mali_conv2d_nchw_winograd_V).x,get_image_dim(mali_conv2d_nchw_winograd_V).y);
    //return;

  if (w_pack_wino >= nw_pack_wino || h_pack_wino >= nh_pack_wino || c_pack4 >= nc_pack4) return;

  const int w = (w_pack_wino << 1) - 1;
  const int h = (h_pack_wino << 1) - 1;

  float4 d00 = h >= 0 ? read_imagef(input, sampler, (int2)((w + 0), (h + 0) + c_pack4 * nh)) : 0;
  float4 d01 = h >= 0 ? read_imagef(input, sampler, (int2)((w + 1), (h + 0) + c_pack4 * nh)) : 0;
  float4 d02 = h >= 0 ? read_imagef(input, sampler, (int2)((w + 2), (h + 0) + c_pack4 * nh)) : 0;
  float4 d03 = h >= 0 ? read_imagef(input, sampler, (int2)((w + 3), (h + 0) + c_pack4 * nh)) : 0;
  float4 d10 = read_imagef(input, sampler, (int2)((w + 0), (h + 1) + c_pack4 * nh));
  float4 d11 = read_imagef(input, sampler, (int2)((w + 1), (h + 1) + c_pack4 * nh));
  float4 d12 = read_imagef(input, sampler, (int2)((w + 2), (h + 1) + c_pack4 * nh));
  float4 d13 = read_imagef(input, sampler, (int2)((w + 3), (h + 1) + c_pack4 * nh));
  float4 d20 = h + 2 < nh ? read_imagef(input, sampler, (int2)((w + 0), (h + 2) + c_pack4 * nh)) : 0;
  float4 d21 = h + 2 < nh ? read_imagef(input, sampler, (int2)((w + 1), (h + 2) + c_pack4 * nh)) : 0;
  float4 d22 = h + 2 < nh ? read_imagef(input, sampler, (int2)((w + 2), (h + 2) + c_pack4 * nh)) : 0;
  float4 d23 = h + 2 < nh ? read_imagef(input, sampler, (int2)((w + 3), (h + 2) + c_pack4 * nh)) : 0;
  float4 d30 = h + 3 < nh ? read_imagef(input, sampler, (int2)((w + 0), (h + 3) + c_pack4 * nh)) : 0;
  float4 d31 = h + 3 < nh ? read_imagef(input, sampler, (int2)((w + 1), (h + 3) + c_pack4 * nh)) : 0;
  float4 d32 = h + 3 < nh ? read_imagef(input, sampler, (int2)((w + 2), (h + 3) + c_pack4 * nh)) : 0;
  float4 d33 = h + 3 < nh ? read_imagef(input, sampler, (int2)((w + 3), (h + 3) + c_pack4 * nh)) : 0;

  float4 z00 = d00 - d02 - d20 + d22;
  float4 z01 = d01 + d02 - d21 - d22;
  float4 z02 = -d01 + d02 + d21 - d22;
  float4 z03 = d01 - d03 - d21 + d23;
  float4 z10 = d10 - d12 + d20 - d22;
  float4 z11 = d11 + d12 + d21 + d22;
  float4 z12 = -d11 + d12 - d21 + d22;
  float4 z13 = d11 - d13 + d21 - d23;
  float4 z20 = -d10 + d12 + d20 - d22;
  float4 z21 = -d11 - d12 + d21 + d22;
  float4 z22 = d11 - d12 - d21 + d22;
  float4 z23 = -d11 + d13 + d21 - d23;
  float4 z30 = d10 - d12 - d30 + d32;
  float4 z31 = d11 + d12 - d31 - d32;
  float4 z32 = -d11 + d12 + d31 - d32;
  float4 z33 = d11 - d13 - d31 + d33;

  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x0 + c_pack4 * 16), z00);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x1 + c_pack4 * 16), z01);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x2 + c_pack4 * 16), z02);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x3 + c_pack4 * 16), z03);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x4 + c_pack4 * 16), z10);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x5 + c_pack4 * 16), z11);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x6 + c_pack4 * 16), z12);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x7 + c_pack4 * 16), z13);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x8 + c_pack4 * 16), z20);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x9 + c_pack4 * 16), z21);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xa + c_pack4 * 16), z22);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xb + c_pack4 * 16), z23);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xc + c_pack4 * 16), z30);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xd + c_pack4 * 16), z31);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xe + c_pack4 * 16), z32);
  write_imagef(input_mat, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xf + c_pack4 * 16), z33);
}

    )";
  } 
  else if (find_M == true && (find_U | find_V ) == false) {
    // only M,inverse output
    this->stream << R"(
//printf("mali_conv2d_nchw_winograd_output %d,%d",get_image_dim(mali_conv2d_nchw_winograd_output).x,get_image_dim(mali_conv2d_nchw_winograd_output).y);
   // return;
        #define output_mat mali_conv2d_nchw_winograd_M 
        #define output mali_conv2d_nchw_winograd_output 
        const int nk_pack4 = (get_image_dim(output).y/get_image_dim(output).x);
        const int nh = get_image_dim(output).x;
        const int nh_pack_wino = (nh+2-1)/2;
        const int nw_pack_wino = nh_pack_wino;


  const int w_pack_wino = get_global_id(0);
  const int h_pack_wino = get_global_id(1);
  const int k_pack4 = get_global_id(2);

  if (w_pack_wino >= nw_pack_wino || h_pack_wino >= nh_pack_wino || k_pack4 >= nk_pack4) return;

  const int w = w_pack_wino << 1;
  const int h = h_pack_wino << 1;

  float4 d00 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x0 + k_pack4 * 16));
  float4 d01 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x1 + k_pack4 * 16));
  float4 d02 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x2 + k_pack4 * 16));
  float4 d03 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x3 + k_pack4 * 16));
  float4 d10 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x4 + k_pack4 * 16));
  float4 d11 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x5 + k_pack4 * 16));
  float4 d12 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x6 + k_pack4 * 16));
  float4 d13 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x7 + k_pack4 * 16));
  float4 d20 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x8 + k_pack4 * 16));
  float4 d21 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0x9 + k_pack4 * 16));
  float4 d22 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xa + k_pack4 * 16));
  float4 d23 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xb + k_pack4 * 16));
  float4 d30 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xc + k_pack4 * 16));
  float4 d31 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xd + k_pack4 * 16));
  float4 d32 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xe + k_pack4 * 16));
  float4 d33 = read_imagef(output_mat, sampler, (int2)((w_pack_wino + h_pack_wino * nw_pack_wino), 0xf + k_pack4 * 16));

  float4 z00 = d00 + d10 + d20 + d01 + d11 + d21 + d02 + d12 + d22;
  float4 z01 = d01 + d11 + d21 - d02 - d12 - d22 - d03 - d13 - d23;
  float4 z10 = d10 - d20 - d30 + d11 - d21 - d31 + d12 - d22 - d32;
  float4 z11 = d11 - d21 - d31 - d12 + d22 + d32 - d13 + d23 + d33;

  write_imagef(output, (int2)((w + 0), (h + 0) + k_pack4 * nh), z00);
  write_imagef(output, (int2)((w + 1), (h + 0) + k_pack4 * nh), z01);
  write_imagef(output, (int2)((w + 0), (h + 1) + k_pack4 * nh), z10);
  write_imagef(output, (int2)((w + 1), (h + 1) + k_pack4 * nh), z11);
}

    )";
  } 
  else if ((find_M & find_U & find_V) == true) {
  // all batch gemm
    this->stream << R"(  

    #define A mali_conv2d_nchw_winograd_U 
    #define B mali_conv2d_nchw_winograd_V 
    #define C mali_conv2d_nchw_winograd_M 
    const int nbatch=4*4;
    const int nm_pack4=get_image_dim(mali_conv2d_nchw_winograd_U).x/4/4;
    const int nk_pack4=get_image_dim(mali_conv2d_nchw_winograd_U).y;
    const int nn_pack4=get_image_dim(mali_conv2d_nchw_winograd_V).x;

    const int n_pack4 = get_global_id(0);
  const int m_pack4 = get_global_id(1);
  const int batch = get_global_id(2);
    //if (m_pack4==0&&batch==0)
    //printf("mali_conv2d_nchw_winograd_M %d,%d....",get_image_dim(mali_conv2d_nchw_winograd_M).x,get_image_dim(mali_conv2d_nchw_winograd_M).y);
    //return;

  if (batch >= nbatch || n_pack4 >= nn_pack4 || m_pack4 >= nm_pack4) return;

  const int n = n_pack4 << 2;

  float4 a0, a1, a2, a3;
  float4 b0, b1, b2, b3;
  float4 c0 = 0, c1 = 0, c2 = 0, c3 = 0;

  for (short k_pack4 = 0; k_pack4 < nk_pack4; k_pack4 += 1) {
    const int k = k_pack4 << 2;

    a0 = read_imagef(A, sampler, (int2)(m_pack4 + batch * nm_pack4, k + 0));
    a1 = read_imagef(A, sampler, (int2)(m_pack4 + batch * nm_pack4, k + 1));
    a2 = read_imagef(A, sampler, (int2)(m_pack4 + batch * nm_pack4, k + 2));
    a3 = read_imagef(A, sampler, (int2)(m_pack4 + batch * nm_pack4, k + 3));

    b0 = read_imagef(B, sampler, (int2)(n + 0, batch + k_pack4 * nbatch));
    b1 = read_imagef(B, sampler, (int2)(n + 1, batch + k_pack4 * nbatch));
    b2 = read_imagef(B, sampler, (int2)(n + 2, batch + k_pack4 * nbatch));
    b3 = read_imagef(B, sampler, (int2)(n + 3, batch + k_pack4 * nbatch));

    c0 = mad(b0.x, a0, c0);
    c0 = mad(b0.y, a1, c0);
    c0 = mad(b0.z, a2, c0);
    c0 = mad(b0.w, a3, c0);

    c1 = mad(b1.x, a0, c1);
    c1 = mad(b1.y, a1, c1);
    c1 = mad(b1.z, a2, c1);
    c1 = mad(b1.w, a3, c1);

    c2 = mad(b2.x, a0, c2);
    c2 = mad(b2.y, a1, c2);
    c2 = mad(b2.z, a2, c2);
    c2 = mad(b2.w, a3, c2);

    c3 = mad(b3.x, a0, c3);
    c3 = mad(b3.y, a1, c3);
    c3 = mad(b3.z, a2, c3);
    c3 = mad(b3.w, a3, c3);
  }

  write_imagef(C, (int2)(n + 0, batch + m_pack4 * nbatch), c0);
  write_imagef(C, (int2)(n + 1, batch + m_pack4 * nbatch), c1);
  write_imagef(C, (int2)(n + 2, batch + m_pack4 * nbatch), c2);
  write_imagef(C, (int2)(n + 3, batch + m_pack4 * nbatch), c3);
    }
    )";
  }
  if (find_M || find_U || find_V) {
    return;
  }
  //===============


  this->PreFunctionBody(f);
  int func_scope = this->BeginScope();
  this->PrintStmt(f->body);
  this->PrintFinalReturn();
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n\n";
}

void CodeGenC::PrintDeclareWithBody(std::function<void()> f) {
  
  std::ostringstream new_stream;
  std::unordered_map<std::string, std::string> new_var_declare_map_;
  new_var_declare_map_.swap(var_declare_map_);
  new_stream.swap(stream);
//PrintStmt(n);
  f();
  new_stream.swap(stream);
  new_var_declare_map_.swap(var_declare_map_);
  for (auto it : new_var_declare_map_) {
    PrintIndent();
    stream << "const int " << it.second << " = " << it.first << ";\n";
  }
  new_var_declare_map_.clear();
  stream << new_stream.str();
  
}

void CodeGenC::PrintFuncPrefix() { stream << "void"; }

void CodeGenC::PrintFinalReturn() {}

std::string CodeGenC::Finish() { return decl_stream.str() + stream.str(); }

void CodeGenC::PrintExpr(const PrimExpr& n, std::ostream& os) {  // NOLINT(*)
  if (print_ssa_form_) {
    std::ostringstream temp;
    VisitExpr(n, temp);
    os << SSAGetID(temp.str(), n.dtype());
  } else {
    VisitExpr(n, os);
  }
}

static bool CheckOutermostBracketMatch(const std::string& s);

void CodeGenC::PrintSSAAssign(const std::string& target, const std::string& src, DataType t) {
  PrintType(t, stream);
  stream << ' ' << target << " = ";
  if (CheckOutermostBracketMatch(src)) {
    stream << src.substr(1, src.length() - 2);
  } else {
    stream << src;
  }
  stream << ";\n";
}

// Print a reference expression to a buffer.
std::string CodeGenC::GetBufferRef(DataType t, const VarNode* buffer, PrimExpr index) {
  std::ostringstream os;
  std::string vid = GetVarID(buffer);
  std::string scope;
  if (alloc_storage_scope_.count(buffer)) {
    scope = alloc_storage_scope_.at(buffer);
  }
  bool is_vol = IsVolatile(buffer);

  if (t.lanes() == 1) {
    if (auto* ptr = index.as<tir::RampNode>()) {
      int64_t lanes = ptr->lanes;
      auto stride = ptr->stride;
      auto base = ptr->base;
      arith::PVar<PrimExpr> basec;
      if (arith::ramp(basec, 1, lanes).Match(index)) {
        PrimExpr lane_int = IntImm(DataType::Int(32), lanes);
        std::ostringstream tmpos;
        tmpos << base;
        std::string strind = Simplify_with_const_var(tmpos.str());
        std::replace(strind.begin(), strind.end(),'.', '_');
        if (strind.find_first_not_of("012345678 ") == std::string::npos) {
          int ind = std::stoi(strind) / lanes;
          os << vid << '[' << ind << ']';
        } else {
          os << vid << '[' << strind << " / " << lane_int << ']';
        }
        return os.str();
      }
    }
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
      os << "*)" << vid << ')';
    } else {
      os << vid;
    }
    os << "[(";
    std::ostringstream tmpos;
    PrintExpr(index, tmpos);
    std::string strind = Simplify_with_const_var(tmpos.str());
    os << strind;
    os << ")";
    if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
      os << " / " << (32 / t.bits());
    }
    os << ']';
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
    std::ostringstream tmpos;
    PrintExpr(index, tmpos);
    std::string strind = Simplify_with_const_var(tmpos.str());
    os << strind;
    os << ")";
    if (t.bits() == 4 || (t.bits() == 1 && t.is_int())) {
      os << " / " << (32 / t.bits());
    }
    os << "))[0]";
  }
  return os.str();
}

// Print a reference expression to a buffer.
std::string CodeGenC::GetStructRef(DataType t, const PrimExpr& buffer, const PrimExpr& index,
                                   int kind) {
  if (kind < builtin::kArrKindBound_) {
    std::ostringstream os;
    os << "(((DLTensor*)";
    this->PrintExpr(buffer, os);
    os << ")";
    if (kind == builtin::kArrAddr) {
      os << " + ";
      this->PrintExpr(index, os);
      os << ")";
      return os.str();
    }
    os << '[';
    this->PrintExpr(index, os);
    os << "].";
    // other case: get fields.
    switch (kind) {
      case builtin::kArrData:
        os << "data";
        break;
      case builtin::kArrShape:
        os << "shape";
        break;
      case builtin::kArrStrides:
        os << "strides";
        break;
      case builtin::kArrNDim:
        os << "ndim";
        break;
      case builtin::kArrTypeCode:
        os << "dtype.code";
        break;
      case builtin::kArrTypeBits:
        os << "dtype.bits";
        break;
      case builtin::kArrByteOffset:
        os << "byte_offset";
        break;
      case builtin::kArrTypeLanes:
        os << "dtype.lanes";
        break;
      case builtin::kArrDeviceId:
        os << "ctx.device_id";
        break;
      case builtin::kArrDeviceType:
        os << "ctx.device_type";
        break;
      default:
        LOG(FATAL) << "unknown field code";
    }
    os << ')';
    return os.str();
  } else {
    ICHECK_LT(kind, builtin::kTVMValueKindBound_);
    std::ostringstream os;
    os << "(((TVMValue*)";
    this->PrintExpr(buffer, os);
    os << ")[" << index << "].";
    if (t.is_handle()) {
      os << "v_handle";
    } else if (t.is_float()) {
      os << "v_float64";
    } else if (t.is_int()) {
      os << "v_int64";
    } else {
      LOG(FATAL) << "Do not know how to handle type" << t;
    }
    os << ")";
    return os.str();
  }
}

bool CodeGenC::HandleTypeMatch(const VarNode* buf_var, DataType t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) return false;
  return it->second == t;
}

void CodeGenC::RegisterHandleType(const VarNode* buf_var, DataType t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_[buf_var] = t;
  } else {
    ICHECK(it->second == t) << "conflicting buf var type";
  }
}

void CodeGenC::PrintVecElemLoad(const std::string& vec, DataType t, int i,
                                std::ostream& os) {  // NOLINT(*)
  os << vec << ".s" << std::hex << i << std::dec;
}

void CodeGenC::PrintVecElemStore(const std::string& vec, DataType t, int i,
                                 const std::string& value) {
  this->PrintIndent();
  stream << vec << ".s" << std::hex << i << " = " << value << ";\n" << std::dec;
}

std::string CodeGenC::GetVecLoad(DataType t, const VarNode* buffer, PrimExpr base) {
  return GetBufferRef(t, buffer, base);
}

void CodeGenC::PrintVecStore(const VarNode* buffer, DataType t, PrimExpr base,
                             const std::string& value) {
  std::string ref = GetBufferRef(t, buffer, base);
  this->PrintIndent();
  stream << ref << " = " << value << ";\n";
}

std::string CodeGenC::CastFromTo(std::string value, DataType from, DataType target) {
  if (from == target) return value;
  std::ostringstream os;
  os << "((";
  this->PrintType(target, os);
  os << ")" << value << ")";
  return os.str();
}

void CodeGenC::BindThreadIndex(const IterVar& iv) { LOG(FATAL) << "not implemented"; }

void CodeGenC::PrintStorageSync(const CallNode* op) {  // NOLINT(*)
}

void CodeGenC::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
  ICHECK_EQ(scope, "global");
}

void CodeGenC::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  ICHECK_EQ(t.lanes(), 1) << "do not yet support vector types";
  if (t.is_handle()) {
    os << "void*";
    return;
  }
  if (t.is_float()) {
    if (t.bits() == 32) {
      os << "float";
      return;
    }
    if (t.bits() == 64) {
      os << "double";
      return;
    }
  } else if (t.is_uint()) {
    switch (t.bits()) {
      case 8:
      case 16:
      case 32:
      case 64: {
        os << "uint" << t.bits() << "_t";
        return;
      }
      case 1:
        os << "int";
        return;
    }
  } else if (t.is_int()) {
    switch (t.bits()) {
      case 8:
      case 16:
      case 32:
      case 64: {
        os << "int" << t.bits() << "_t";
        return;
      }
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

void CodeGenC::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    std::ostringstream ss;
    PrintType(ptr->element_type, ss);
    std::string printstr = ss.str();
    os << printstr;
    if (printstr == "image2d_t") {
      return;
    }
    os << '*';
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}

inline void PrintConst(const IntImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
  if (op->dtype == DataType::Int(32)) {
    std::ostringstream temp;
    temp << op->value;
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(op->dtype, os);
    os << ")" << op->value;
  }
}

inline void PrintUIntConst(DataType dtype, uint64_t val, std::ostream& os,
                           CodeGenC* p) {  // NOLINT(*)
  if (dtype == DataType::UInt(32)) {
    std::ostringstream temp;
    temp << val << "U";
    p->MarkConst(temp.str());
    os << temp.str();
  } else {
    os << "(";
    p->PrintType(dtype, os);
    os << ")" << val;
  }
}

inline void PrintConst(const FloatImmNode* op, std::ostream& os, CodeGenC* p) {  // NOLINT(*)
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
      std::ostringstream temp;
      temp << std::scientific << op->value;
      if (op->dtype.bits() == 32) temp << 'f';
      p->MarkConst(temp.str());
      os << temp.str();
      break;
    }
    case 16: {
      os << '(';
      p->PrintType(op->dtype, os);
      os << ')' << std::scientific << op->value << 'f';
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

void CodeGenC::VisitExpr_(const IntImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}

void CodeGenC::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  PrintConst(op, os, this);
}
void CodeGenC::VisitExpr_(const StringImmNode* op, std::ostream& os) {  // NOLINT(*)
  os << "\"" << op->value << "\"";
}

template <typename T>
inline void PrintBinaryExpr(const T* op, const char* opstr,
                            std::ostream& os,  // NOLINT(*)
                            CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
    if (isalpha(opstr[0])) {
      os << opstr << '(';
      p->PrintExpr(op->a, os);
      os << ", ";
      p->PrintExpr(op->b, os);
      os << ')';
    } else {
      os << '(';
      p->PrintExpr(op->a, os);
      os << ' ' << opstr << ' ';
      p->PrintExpr(op->b, os);
      os << ')';
    }
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->a, op->b, os);
  }
}

inline void PrintBinaryIntrinsic(const CallNode* op, const char* opstr,
                                 std::ostream& os,  // NOLINT(*)
                                 CodeGenC* p) {
  if (op->dtype.lanes() == 1) {
    ICHECK_EQ(op->args.size(), 2U);
    os << '(';
    p->PrintExpr(op->args[0], os);
    os << opstr;
    p->PrintExpr(op->args[1], os);
    os << ')';
  } else {
    p->PrintVecBinaryOp(opstr, op->dtype, op->args[0], op->args[1], os);
  }
}
void CodeGenC::VisitExpr_(const CastNode* op, std::ostream& os) {  // NOLINT(*)
  std::stringstream value;
  this->PrintExpr(op->value, value);
  os << CastFromTo(value.str(), op->value.dtype(), op->dtype);
}
void CodeGenC::VisitExpr_(const VarNode* op, std::ostream& os) {  // NOLINT(*)
  os << GetVarID(op);
}
void CodeGenC::VisitExpr_(const AddNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "+", os, this);
}
void CodeGenC::VisitExpr_(const SubNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "-", os, this);
}
void CodeGenC::VisitExpr_(const MulNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "*", os, this);
}
void CodeGenC::VisitExpr_(const DivNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "/", os, this);
}
void CodeGenC::VisitExpr_(const ModNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "%", os, this);
}
void CodeGenC::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "min", os, this);
}
void CodeGenC::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "max", os, this);
}
void CodeGenC::VisitExpr_(const EQNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "==", os, this);
}
void CodeGenC::VisitExpr_(const NENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "!=", os, this);
}
void CodeGenC::VisitExpr_(const LTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<", os, this);
}
void CodeGenC::VisitExpr_(const LENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "<=", os, this);
}
void CodeGenC::VisitExpr_(const GTNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">", os, this);
}
void CodeGenC::VisitExpr_(const GENode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, ">=", os, this);
}
void CodeGenC::VisitExpr_(const AndNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "&&", os, this);
}
void CodeGenC::VisitExpr_(const OrNode* op, std::ostream& os) {  // NOLINT(*)
  PrintBinaryExpr(op, "||", os, this);
}
void CodeGenC::VisitExpr_(const NotNode* op, std::ostream& os) {  // NOLINT(*)
  os << '!';
  PrintExpr(op->a, os);
}

void removeSubstrs(std::string& s, const std::string& p) {
  std::string::size_type n = p.length();

  for (std::string::size_type i = s.find(p); i != std::string::npos;
       i = s.find(p))
    s.erase(i, n);
}
void CodeGenC::PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                               bool skip_first_arg, std::ostream& os) {  // NOLINT(*)
    //read_image or write_image axis func
    if (global_symbol == "image_axis") {
      
      os << "(int2)(";
      ICHECK_EQ(skip_first_arg, true) << " can't skip_first_arg";
      ICHECK_EQ(args.size(), 3) << " args for image axis must be 2";
      //DataType dtype = args[1].dtype();
      //ICHECK_EQ(dtype.lanes(), 4) << " only rgba is support yet ";
      for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
        PrimExpr tmp_prim = args[i];
        if (auto* ptr = tmp_prim.as<tir::RampNode>()) {
          ICHECK_EQ(ptr->lanes, 4) << " only rgba is support yet ";
          std::ostringstream temp;
          this->PrintExpr(ptr->base, temp);
          // 4 == lanes
          temp << "/4";
          std::string img_x_axes = tvm::tir::exprSimp::DoSimplify(temp.str());
          img_x_axes = Simplify_with_const_var(img_x_axes);
          os << img_x_axes;
        } else if (auto* ptr = tmp_prim.as<tir::BroadcastNode>()) {
          std::ostringstream temp;
          this->PrintExpr(ptr->value, temp);
          std::string img_y_axes = Simplify_with_const_var(temp.str());
          os << img_y_axes;
        }
        else {
          ICHECK(false) << " should not happen";
        }
        if (i < args.size() - 1) {
          os << ", ";
        }
      }
      os << ")";
      return;
  }
  if (global_symbol != "mul_must_be_replace") {
    os << global_symbol << "(";
  }
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    this->PrintExpr(args[i], os);
    if (i < args.size() - 1) {
      os << ", ";
    }
  }
  if (global_symbol != "mul_must_be_replace") {
    os << ")";
  }
}

void CodeGenC::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr_op = op->op.as<OpNode>()) {
    auto call_op = GetRef<Op>(ptr_op);

    if (op->op.same_as(builtin_call_extern_) || op->op.same_as(builtin_call_pure_extern_)) {
      ICHECK_GE(op->args.size(), 1U);
      auto func = Downcast<StringImm>(op->args[0]);
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), func->value, op->args, true, os);
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      this->PrintCallExtern(GetType(GetRef<PrimExpr>(op)), op_attr_global_symbol_[call_op],
                            op->args, false, os);
    } else if (op->op.same_as(builtin::bitwise_and())) {
      PrintBinaryIntrinsic(op, " & ", os, this);
    } else if (op->op.same_as(builtin::large_uint_imm())) {
      ICHECK_EQ(op->args.size(), 2U);
      uint64_t low = static_cast<uint64_t>(Downcast<IntImm>(op->args[0])->value);
      uint64_t high = static_cast<uint64_t>(Downcast<IntImm>(op->args[1])->value);
      uint64_t val = (high << 32U) | low;
      PrintUIntConst(op->dtype, val, os, this);
    } else if (op->op.same_as(builtin::bitwise_xor())) {
      PrintBinaryIntrinsic(op, " ^ ", os, this);
    } else if (op->op.same_as(builtin::bitwise_or())) {
      PrintBinaryIntrinsic(op, " | ", os, this);
    } else if (op->op.same_as(builtin::bitwise_not())) {
      ICHECK_EQ(op->args.size(), 1U);
      os << "(~";
      this->PrintExpr(op->args[0], os);
      os << ')';
    } else if (op->op.same_as(builtin::shift_left())) {
      PrintBinaryIntrinsic(op, " << ", os, this);
    } else if (op->op.same_as(builtin::shift_right())) {
      PrintBinaryIntrinsic(op, " >> ", os, this);
    } else if (op->op.same_as(builtin::if_then_else())) {
      os << "(";
      PrintExpr(op->args[0], os);
      os << " ? ";
      PrintExpr(op->args[1], os);
      os << " : ";
      PrintExpr(op->args[2], os);
      os << ")";
    } else if (op->op.same_as(builtin::address_of())) {
      const LoadNode* l = op->args[0].as<LoadNode>();
      ICHECK(op->args.size() == 1 && l);
      os << "((";
      this->PrintType(l->dtype.element_of(), os);
      os << " *)" << this->GetVarID(l->buffer_var.get()) << " + "
         << "(";
      this->PrintExpr(l->index, os);
      if (l->dtype.bits() == 4 || (l->dtype.bits() == 1 && l->dtype.is_int())) {
        os << " / " << (32 / l->dtype.bits());
      }
      os << "))";
    } else if (op->op.same_as(builtin::tvm_struct_get())) {
      ICHECK_EQ(op->args.size(), 3U);
      os << GetStructRef(op->dtype, op->args[0], op->args[1], op->args[2].as<IntImmNode>()->value);
    } else if (op->op.same_as(builtin::isnullptr())) {
      ICHECK_EQ(op->args.size(), 1U);
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " == NULL)";
    } else if (op->op.same_as(builtin::reinterpret())) {
      int ssa_scope = BeginScope();
      std::string rhs = SSAGetID(PrintExpr(op->args[0]), op->args[0]->dtype);
      os << "(*(";
      this->PrintType(op->dtype, os);
      os << " *)(&(" << rhs << ")))";
      EndScope(ssa_scope);
    } else if (op->op.same_as(builtin::isnan())) {
      os << "(";
      this->PrintExpr(op->args[0], os);
      os << " != ";
      this->PrintExpr(op->args[0], os);
      os << ")";
    } else {
      LOG(FATAL) << "Unresolved call " << op->op;
    }
  } else {
    ICHECK(op->op.as<GlobalVarNode>());
    LOG(FATAL) << "Do not yet support cross function call";
  }
}

void CodeGenC::PrintVecBinaryOp(const std::string& op, DataType t, PrimExpr lhs, PrimExpr rhs,
                                std::ostream& os) {  // NOLINT(*)
  if (isalpha(op[0])) {
    os << op << "(";
    this->PrintExpr(lhs, os);
    os << ", ";
    this->PrintExpr(rhs, os);
    os << ")";
  } else {
    os << "(";
    this->PrintExpr(lhs, os);
    os << ' ' << op << ' ';
    this->PrintExpr(rhs, os);
    os << ")";
  }
}

void CodeGenC::VisitExpr_(const LoadNode* op, std::ostream& os) {  // NOLINT(*)
  int lanes = op->dtype.lanes();
  std::string vid = GetVarID(op->buffer_var.get());
  // delcare type.
  if (op->dtype.lanes() == 1) {
    const VarNode* buffer = op->buffer_var.get();
    std::string ref = GetBufferRef(op->dtype, buffer, op->index);
    std::string scope;
    if (alloc_storage_scope_.count(buffer)) {
      scope = alloc_storage_scope_.at(buffer);
    }
    if (scope.compare(0, sizeof("image")-1, "image") == 0) {
      ref = "read_imagef(" + vid + ",sampler," + ref + ").x";
    }
    HandleVolatileLoads(ref, op, os);
  } else {
    ICHECK(is_one(op->predicate)) << "predicated load is not supported";

    arith::PVar<PrimExpr> base;
    bool is_climg_bug_not_cl_rgba = op->dtype.is_climgfloatrw() && (USE_CL_RGBA == 0);
    if (is_climg_bug_not_cl_rgba == false) {
      if (arith::ramp(base, 1, op->dtype.lanes()).Match(op->index)) {
        std::string ref = GetVecLoad(op->dtype, op->buffer_var.get(), base.Eval());
        HandleVolatileLoads(ref, op, os);
      } else if (auto *axis_p = (op->index).as<CallNode>()) {
        std::string ref = GetVecLoad(op->dtype, op->buffer_var.get(), op->index);
        //std::string ref = PrintExpr(op->index);
        HandleVolatileLoads(ref, op, os);
      }
      else {
        ICHECK(false) << " should go here";
      }
    } else {
      std::ostringstream svalue_expr;
      std::string sindex = SSAGetID(PrintExpr(op->index), op->index.dtype());
      
      DataType elem_type = op->dtype.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (op->buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          if (elem_type.is_climgfloatrw()) {
            value_temp << "read_imagef(" << vid << ",sampler,";
          } else {
            value_temp << vid;
          }
        }
        std::ostringstream index_temp;
        PrintVecElemLoad(sindex, op->index.dtype(), i, index_temp);
        if (elem_type.is_climgfloatrw()) {
#if USE_CL_RGBA
          LOG(FATAL) << "Not support read 1 element from 4-channel image";
#else 
          value_temp << get_2Dmemo_floatx1_int2(vid, index_temp.str()) << ").x";
#endif
        } else {
          value_temp << '[';
          value_temp << index_temp.str();
          value_temp << ']';
        }
        PrintVecElemLoadExpr(op->dtype, i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

void CodeGenC::VisitStmt_(const StoreNode* op) {
  std::string vid = GetVarID(op->buffer_var.get());
  DataType t = op->value.dtype();
  bool is_climg =
      alloc_storage_scope_.count(op->buffer_var.get()) &&
      alloc_storage_scope_[op->buffer_var.get()].compare(0, sizeof("image") - 1, "image") == 0;
  if (t.lanes() == 1) {
    std::string value = this->PrintExpr(op->value);
    std::string ref = this->GetBufferRef(t, op->buffer_var.get(), op->index);
    this->PrintIndent();

    // read_image
    // write_image
    if (is_climg) {
#if USE_CL_RGBA
      //LOG(FATAL) << "floatx4 Image store is not supported here";
#else
      Store_2Dmemo_floatx1(vid, ref, value);
#endif
    } else {
      stream << ref << " = " << value << ";\n";
    }
  } else {
    ICHECK(is_one(op->predicate)) << "Predicated store is not supported";
    arith::PVar<PrimExpr> base;

   
    bool is_climg_bug_not_cl_rgba = is_climg && (USE_CL_RGBA == 0);
    if (is_climg_bug_not_cl_rgba == false) {
      if (arith::ramp(base, 1, t.lanes()).Match(op->index)) {
        std::string value = this->PrintExpr(op->value);
        auto new_code = is_climg ? DataType::kCLImgFloatW : DataType::kFloat;
        this->PrintVecStore(op->buffer_var.get(), t.with_code(new_code), base.Eval(), value);
      } else if (auto* axis_p = (op->index).as<CallNode>()) {
        auto new_code = is_climg ? DataType::kCLImgFloatW : DataType::kFloat;
        std::string value = this->PrintExpr(op->value);
        this->PrintVecStore(op->buffer_var.get(), t.with_code(new_code), op->index, value);
      } else {
      
      ICHECK(false) << "should not go here";
      }
    } else {
      // The assignment below introduces side-effect, and the resulting value cannot
      // be reused across multiple expression, thus a new scope is needed
      int vec_scope = BeginScope();

      // store elements seperately
      std::string index =
          SSAGetID(PrintExpr(op->index), op->index.dtype());
      std::string value = SSAGetID(PrintExpr(op->value), op->value.dtype());
      for (int i = 0; i < t.lanes(); ++i) {
        this->PrintIndent();
        DataType elem_type = t.element_of();
        if (!HandleTypeMatch(op->buffer_var.get(), elem_type)) {
          stream << "((";
          if (op->buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(op->buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, stream);
            }
          }
          PrintType(elem_type, stream);
          stream << "*)" << vid << ')';
        } else {
          if (!is_climg) {
            stream << vid;
          }
        }
        if (is_climg) {
          std::ostringstream elmoss;
          PrintVecElemLoad(index, op->index.dtype(), i, elmoss);
          std::string ref = elmoss.str();
          elmoss.str("");
          elmoss.clear();
          PrintVecElemLoad(value, op->value.dtype(), i, elmoss);
          Store_2Dmemo_floatx1(vid, ref, elmoss.str());
        } else {
          stream << '[';
          PrintVecElemLoad(index, op->index.dtype(), i, stream);
          stream << "] = ";
          PrintVecElemLoad(value, op->value.dtype(), i, stream);
          stream << ";\n";
        }
      }
      EndScope(vec_scope);
    }
  }
}

void CodeGenC::VisitExpr_(const LetNode* op, std::ostream& os) {  // NOLINT(*)
  auto it = let_binding_.find(op->var);
  if (it != let_binding_.end()) {
    ICHECK(deep_equal_(it->second->value, op->value))
        << "Let cannot bind the same var to two different values";
  } else {
    let_binding_[op->var] = op;
  }
  std::string value = PrintExpr(op->value);
  var_idmap_[op->var.get()] = value;
  os << PrintExpr(op->body);
}

void CodeGenC::VisitExpr_(const RampNode* op, std::ostream& os) {  // NOLINT(*)
  // constraint of current logic
  ICHECK_EQ(op->base.dtype(), DataType::Int(32));
  os << "((int" << op->lanes << ")(";
  for (int i = 0; i < op->lanes; i++) {
    std::ostringstream subexpr;
    subexpr << "(" << Simplify_with_const_var(PrintExpr(op->base)) << ")"
       << "+(" << PrintExpr(op->stride) << "*" << i << ")";
    os << tvm::tir::exprSimp::DoSimplify(subexpr.str());
    if (i != op->lanes - 1) os << ", ";
  }
  os << "))";
}

void CodeGenC::VisitExpr_(const ShuffleNode* op, std::ostream& os) {
  LOG(FATAL) << "Shuffle: not supported ";
}

void CodeGenC::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  LOG(FATAL) << "Broadcast: not supported ";
}

void CodeGenC::VisitExpr_(const SelectNode* op, std::ostream& os) {  // NOLINT(*)
  os << "(";
  PrintExpr(op->condition, os);
  os << " ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ")";
}

void CodeGenC::VisitStmt_(const LetStmtNode* op) {
  std::string value = PrintExpr(op->value);
  if (print_ssa_form_) {
    ICHECK(!var_idmap_.count(op->var.get()));
    var_idmap_[op->var.get()] = value;
  } else {
    PrintIndent();
    if (op->var.dtype() == DataType::Handle() && handle_data_type_.count(op->var.get())) {
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "* " << AllocVarID(op->var.get()) << " = (";
      PrintType(handle_data_type_.at(op->var.get()), stream);
      stream << "*)" << value << ";\n";
    } else {
      PrintType(op->var.dtype(), this->stream);
      this->stream << ' ' << AllocVarID(op->var.get()) << " = " << value << ";\n";
    }
  }
  PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AllocateNode* op) {
  ICHECK(!is_zero(op->condition));
  std::string vid = AllocVarID(op->buffer_var.get());

  this->PrintIndent();
  int32_t constant_size = op->constant_allocation_size();
  ICHECK_GT(constant_size, 0) << "Can only handle constant size stack allocation for now";
  const VarNode* buffer = op->buffer_var.as<VarNode>();
  std::string scope = alloc_storage_scope_.at(buffer);
  PrintStorageScope(scope, stream);
  PrintType(op->dtype, stream);
  stream << ' ' << vid << '[' << constant_size << "];\n";

  RegisterHandleType(op->buffer_var.get(), op->dtype);
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == tir::attr::thread_extent) {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag.length() != 0) {
      if (!var_idmap_.count(iv->var.get())) {
        BindThreadIndex(iv);
      }
    }
  } else if (op->attr_key == tir::attr::storage_scope) {
    const VarNode* v = op->node.as<VarNode>();
    ICHECK(v);
    alloc_storage_scope_[v] = op->value.as<StringImmNode>()->value;
  } else if (op->attr_key == tir::attr::volatile_scope) {
    const VarNode* v = op->node.as<VarNode>();
    ICHECK(v);
    volatile_buf_.insert(v);
  } else if (op->attr_key == tir::attr::pragma_import_c) {
    const StringImmNode* value = op->value.as<StringImmNode>();
    ICHECK(value != nullptr);
    decl_stream << value->value;
  }
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const AssertStmtNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (const auto* str = op->message.as<StringImmNode>()) {
    // GLOG style check
    stream << "ICHECK(" << cond << ") << \"" << str->value << "\";\n";
  } else {
    stream << "assert(" << cond << ");\n";
  }
  this->PrintStmt(op->body);
}

void CodeGenC::VisitStmt_(const ForNode* op) {
  std::string extent = PrintExpr(op->extent);
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  ICHECK(is_zero(op->min));
  stream << "for (";
  PrintType(op->loop_var.dtype(), stream);
  stream << ' ' << vid << " = 0; " << vid << " < " << extent << "; ++" << vid << ") {\n";
  int for_scope = BeginScope();
  //===========================
  PrintDeclareWithBody([this, op]() { PrintStmt(op->body); });
  //========================
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::VisitStmt_(const IfThenElseNode* op) {
  std::string cond = PrintExpr(op->condition);
  PrintIndent();
  if (cond[0] == '(' && cond[cond.length() - 1] == ')') {
    stream << "if " << cond << " {\n";
  } else {
    stream << "if (" << cond << ") {\n";
  }
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);

  if (op->else_case.defined()) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case);
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}

void CodeGenC::VisitStmt_(const SeqStmtNode* op) {
  auto print_seq = [this, op]() {
    for (Stmt stmt : op->seq) {
      PrintStmt(stmt);
    }
  };
  PrintDeclareWithBody(print_seq);
}

void CodeGenC::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;
  const CallNode* call = op->value.as<CallNode>();
  if (call) {
    if (call->op.same_as(builtin::tvm_storage_sync())) {
      this->PrintStorageSync(call);
      return;
    } else if (call->op.same_as(builtin::tvm_struct_set())) {
      ICHECK_EQ(call->args.size(), 4);
      std::string value = PrintExpr(call->args[3]);
      std::string ref = GetStructRef(call->args[3].dtype(), call->args[0], call->args[1],
                                     call->args[2].as<IntImmNode>()->value);
      this->PrintIndent();
      this->stream << ref << " = " << value << ";\n";
      return;
    }
  }
  std::string vid = this->PrintExpr(op->value);
  if (vid != "") {
    this->PrintIndent();
    this->stream << "(void)" << vid << ";\n";
  }
}

void CodeGenC::PrintVecElemLoadExpr(DataType t, int i, const std::string& value, std::ostream& os) {
  ICHECK_GT(t.lanes(), 1);
  if (t.bits() == 8 && (t.is_int() || t.is_uint())) {
    if (i != 0) {
      os << "|";
    }
    os << "((0x000000ff << " << i * 8 << ") & (" << value << " << " << i * 8 << "))";
    return;
  }

  if (i == 0) {
    os << "((";
    PrintType(t, os);
    os << ")(";
  }
  os << value;
  if (i != t.lanes() - 1) {
    os << ",";
  } else {
    os << "))";
  }
  return;
}

static bool CheckOutermostBracketMatch(const std::string& s) {
  if (!s.empty() && s.front() == '(' && s.back() == ')') {
    size_t len = s.size();
    int n_unmatched = 0;
    for (size_t i = 0; i < len; ++i) {
      if (s[i] == '(') {
        n_unmatched++;
      } else if (s[i] == ')') {
        n_unmatched--;
      }
      if (n_unmatched == 0) {
        return i == len - 1;
      }
    }
  }
  return false;
}

// for string delimiter
std::vector<std::string> split_string(std::string s, std::string delimiter) {
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

void trimSpace(std::string& s) {
  size_t index = 0;
  if (!s.empty()) {
    while ((index = s.find(' ', index)) != std::string::npos) {
      s.erase(index, 1);
    }
  }
}

//avoid calculate the common index multi-times, especially IFLOPS is extremely lower in armgpu
bool CodeGenC::Find_longst_common_str_or_add_key(const std::string& base,
                                                      std::string& new_base_index) {
  
  auto count_any_of = [](const std::string& src, const std::string& count_charactors) -> int32_t {
    int32_t ans = 0;
    for (auto c : src) {
      if (count_charactors.find(c) != std::string::npos) {
        ans++;
      }
    }
    return ans;
  };
  // for some case, we just skip it to save register
  // 1. constant expr
  // 2. already  referenced common_expr
  // 3. less than two op
  const std::string common_op = "+*-/%";
  if (base.find_first_of(common_op) == std::string::npos ||
      base.find("const_common_") != std::string::npos || count_any_of(base, (common_op)) < 3) {
    new_base_index = base;
    return false;
  }

  std::string remove_space = base;
  trimSpace(remove_space);
  if (var_declare_map_.count(remove_space)) {
    new_base_index = var_declare_map_[remove_space];
    return true;
  }
  std::vector<std::string> vec_base = split_string(remove_space, "+*-/%");
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
  new_base_index = "const_common_" + std::to_string(hash_str(base));
  var_declare_map_[remove_space] = new_base_index;
  return false;
}

std::string CodeGenC::Simplify_with_const_var(const std::string& base) {
  std::string new_base_index = base;
  Find_longst_common_str_or_add_key(base, new_base_index);
  return new_base_index;
}

}  // namespace codegen
}  // namespace tvm
