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
 * \file intrin_rule_llvm.h
 * \brief Common utilities for llvm intrinsics.
 */
#ifndef TVM_TARGET_LLVM_INTRIN_RULE_LLVM_H_
#define TVM_TARGET_LLVM_INTRIN_RULE_LLVM_H_
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/registry.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>

#include <string>

#include "llvm_common.h"

namespace tvm {
namespace codegen {
// num_signature means number of arguments used to query signature
template <unsigned id, int num_signature>
inline void DispatchLLVMPureIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(DataType::UInt(32), id));
  cargs.push_back(IntImm(DataType::UInt(32), num_signature));

  for (PrimExpr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = tir::Call(call->dtype, tir::builtin::call_llvm_pure_intrin(), cargs);
}

template <unsigned id, int num_signature>
inline void DispatchLLVMIntrin(const TVMArgs& targs, TVMRetValue* rv) {
  PrimExpr e = targs[0];
  const tir::CallNode* call = e.as<tir::CallNode>();
  ICHECK(call != nullptr);
  Array<PrimExpr> cargs;
  // intrin id.
  cargs.push_back(IntImm(DataType::UInt(32), id));
  cargs.push_back(IntImm(DataType::UInt(32), num_signature));
  for (PrimExpr arg : call->args) {
    cargs.push_back(arg);
  }
  *rv = tir::Call(call->dtype, tir::builtin::call_llvm_intrin(), cargs);
}

// Return the intrinsic name
struct Direct {
  std::string operator()(DataType t, std::string name) const { return name; }
};

// Call pure extern function.
template <typename T>
inline void DispatchPureExternLLVM(const TVMArgs& args, TVMRetValue* rv) {
  using namespace tir;
  PrimExpr e = args[0];
  const CallNode* call = e.as<CallNode>();
  ICHECK(call != nullptr);
  // Use string based dispatch to extern for backward compact
  // TODO(tvm-team) replace once the new dispatching system is inplace.
  const OpNode* op = call->op.as<OpNode>();
  ICHECK(op != nullptr);
  std::string name = op->name;
  ICHECK_EQ(name.substr(0, 4), "tir.");
  name = T()(call->dtype, name.substr(4));

  if (name.length() != 0) {
    Array<PrimExpr> new_args = {StringImm(name)};
    for (auto arg : call->args) {
      new_args.push_back(arg);
    }
    *rv = Call(call->dtype, tir::builtin::call_pure_extern(), new_args);
  } else {
    *rv = e;
  }
}

}  // namespace codegen
}  // namespace tvm

#endif  // LLVM_VERSION
#endif  // TVM_TARGET_LLVM_INTRIN_RULE_LLVM_H_
