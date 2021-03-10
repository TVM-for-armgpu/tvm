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
 * \file tvm/runtime/registry.h
 * \brief This file defines the TVM global function registry.
 *
 *  The registered functions will be made available to front-end
 *  as well as backend users.
 *
 *  The registry stores type-erased functions.
 *  Each registered function is automatically exposed
 *  to front-end language(e.g. python).
 *
 *  Front-end can also pass callbacks as PackedFunc, or register
 *  then into the same global registry in C++.
 *  The goal is to mix the front-end language and the TVM back-end.
 *
 * \code
 *   // register the function as MyAPIFuncName
 *   TVM_REGISTER_GLOBAL(MyAPIFuncName)
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *     // my code.
 *   });
 * \endcode
 */
#ifndef TVM_RUNTIME_REGISTRY_H_
#define TVM_RUNTIME_REGISTRY_H_

#include <tvm/runtime/packed_func.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief Registry for global function */
class Registry {
 public:
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  TVM_DLL Registry& set_body(PackedFunc f);  // NOLINT(*)
  /*!
   * \brief set the body of the function to be f
   * \param f The body of the function.
   */
  Registry& set_body(PackedFunc::FType f) {  // NOLINT(*)
    return set_body(PackedFunc(f));
  }
  /*!
   * \brief set the body of the function to the given function.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * int multiply(int x, int y) {
   *   return x * y;
   * }
   *
   * TVM_REGISTER_GLOBAL("multiply")
   * .set_body_typed(multiply); // will have type int(int, int)
   *
   * // will have type int(int, int)
   * TVM_REGISTER_GLOBAL("sub")
   * .set_body_typed([](int a, int b) -> int { return a - b; });
   *
   * \endcode
   *
   * \param f The function to forward to.
   * \tparam FLambda The signature of the function.
   */
  template <typename FLambda>
  Registry& set_body_typed(FLambda f) {
    using FType = typename detail::function_signature<FLambda>::FType;
    return set_body(TypedPackedFunc<FType>(std::move(f)).packed());
  }
  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // node subclass:
   * struct Example {
   *    int doThing(int x);
   * }
   * TVM_REGISTER_GLOBAL("Example_doThing")
   * .set_body_method(&Example::doThing); // will have type int(Example, int)
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam T the type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename T, typename R, typename... Args>
  Registry& set_body_method(R (T::*f)(Args...)) {
    auto fwrap = [f](T target, Args... params) -> R {
      // call method pointer
      return (target.*f)(params...);
    };
    return set_body(TypedPackedFunc<R(T, Args...)>(fwrap));
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // node subclass:
   * struct Example {
   *    int doThing(int x);
   * }
   * TVM_REGISTER_GLOBAL("Example_doThing")
   * .set_body_method(&Example::doThing); // will have type int(Example, int)
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam T the type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename T, typename R, typename... Args>
  Registry& set_body_method(R (T::*f)(Args...) const) {
    auto fwrap = [f](const T target, Args... params) -> R {
      // call method pointer
      return (target.*f)(params...);
    };
    return set_body(TypedPackedFunc<R(const T, Args...)>(fwrap));
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Used when calling a method on a Node subclass through a ObjectRef subclass.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // node subclass:
   * struct ExampleNode: BaseNode {
   *    int doThing(int x);
   * }
   *
   * // noderef subclass
   * struct Example;
   *
   * TVM_REGISTER_GLOBAL("Example_doThing")
   * .set_body_method<Example>(&ExampleNode::doThing); // will have type int(Example, int)
   *
   * // note that just doing:
   * // .set_body_method(&ExampleNode::doThing);
   * // wouldn't work, because ExampleNode can't be taken from a TVMArgValue.
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam TObjectRef the node reference type to call the method on
   * \tparam TNode the node type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename TObjectRef, typename TNode, typename R, typename... Args,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  Registry& set_body_method(R (TNode::*f)(Args...)) {
    auto fwrap = [f](TObjectRef ref, Args... params) {
      TNode* target = ref.operator->();
      // call method pointer
      return (target->*f)(params...);
    };
    return set_body(TypedPackedFunc<R(TObjectRef, Args...)>(fwrap));
  }

  /*!
   * \brief set the body of the function to be the passed method pointer.
   *        Used when calling a method on a Node subclass through a ObjectRef subclass.
   *        Note that this will ignore default arg values and always require all arguments to be
   * provided.
   *
   * \code
   *
   * // node subclass:
   * struct ExampleNode: BaseNode {
   *    int doThing(int x);
   * }
   *
   * // noderef subclass
   * struct Example;
   *
   * TVM_REGISTER_GLOBAL("Example_doThing")
   * .set_body_method<Example>(&ExampleNode::doThing); // will have type int(Example, int)
   *
   * // note that just doing:
   * // .set_body_method(&ExampleNode::doThing);
   * // wouldn't work, because ExampleNode can't be taken from a TVMArgValue.
   *
   * \endcode
   *
   * \param f the method pointer to forward to.
   * \tparam TObjectRef the node reference type to call the method on
   * \tparam TNode the node type containing the method (inferred).
   * \tparam R the return type of the function (inferred).
   * \tparam Args the argument types of the function (inferred).
   */
  template <typename TObjectRef, typename TNode, typename R, typename... Args,
            typename = typename std::enable_if<std::is_base_of<ObjectRef, TObjectRef>::value>::type>
  Registry& set_body_method(R (TNode::*f)(Args...) const) {
    auto fwrap = [f](TObjectRef ref, Args... params) {
      const TNode* target = ref.operator->();
      // call method pointer
      return (target->*f)(params...);
    };
    return set_body(TypedPackedFunc<R(TObjectRef, Args...)>(fwrap));
  }

  /*!
   * \brief Register a function with given name
   * \param name The name of the function.
   * \param override Whether allow oveeride existing function.
   * \return Reference to theregistry.
   */
  TVM_DLL static Registry& Register(const std::string& name, bool override = false);  // NOLINT(*)
  /*!
   * \brief Erase global function from registry, if exist.
   * \param name The name of the function.
   * \return Whether function exist.
   */
  TVM_DLL static bool Remove(const std::string& name);
  /*!
   * \brief Get the global function by name.
   * \param name The name of the function.
   * \return pointer to the registered function,
   *   nullptr if it does not exist.
   */
  TVM_DLL static const PackedFunc* Get(const std::string& name);  // NOLINT(*)
  /*!
   * \brief Get the names of currently registered global function.
   * \return The names
   */
  TVM_DLL static std::vector<std::string> ListNames();

  // Internal class.
  struct Manager;

 protected:
  /*! \brief name of the function */
  std::string name_;
  /*! \brief internal packed function */
  PackedFunc func_;
  friend struct Manager;
};

#define TVM_FUNC_REG_VAR_DEF static TVM_ATTRIBUTE_UNUSED ::tvm::runtime::Registry& __mk_##TVM

/*!
 * \brief Register a function globally.
 * \code
 *   TVM_REGISTER_GLOBAL("MyPrint")
 *   .set_body([](TVMArgs args, TVMRetValue* rv) {
 *   });
 * \endcode
 */
#define TVM_REGISTER_GLOBAL(OpName) \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) = ::tvm::runtime::Registry::Register(OpName)

#define TVM_STRINGIZE_DETAIL(x) #x
#define TVM_STRINGIZE(x) TVM_STRINGIZE_DETAIL(x)
#define TVM_DESCRIBE(...) describe(__VA_ARGS__ "\n\nFrom:" __FILE__ ":" TVM_STRINGIZE(__LINE__))
/*!
 * \brief Macro to include current line as string
 */
#define TVM_ADD_FILELINE "\n\nDefined in " __FILE__ ":L" TVM_STRINGIZE(__LINE__)

template <class T>
inline std::vector<T> decode_shape_fold(uint64_t encoded_shape) {
  size_t ndim = encoded_shape >> 56;
  ICHECK((ndim > 0) && (ndim < 7)) << " decode failed, (ndim > 0) && (ndim < 7):vs " << ndim;
  uint64_t full = 0xffffffffffffffff;
  encoded_shape &= (full >> (64 - 56));
  std::vector<T> values(ndim, 0);
    //0---63bit---  w is used for winograd w*h
    //|---bytes---|-shape dim-|----N----|----C------|----H----|----W-----|-----C-----|
    //|-----4-----|-----3-----|----4----|-----16----|---16----|----16----|-----4----|
    
    //|---bytes---|-shape dim-|---I-----|----O------|----H----|----W-----|------1----|-----C-----|
    //|-----4-----|-----3-----|---16----|-----16----|----8----|----8-----|------4----|------4----|

  // 1-4 dim
  //|---bytes---|-shape dim-|---I-----|----O------|----H----|----W-----|
  //|-----4-----|-----3-----|---14----|-----14----|----14---|----14----|
  if (ndim < 3) {
    for(size_t i=0;i<ndim;++i){
      int abc = (28*(values.size()-i-1));
      values[i] = encoded_shape >> abc;
      encoded_shape &= (full >> (64 - abc));
    }
  } else if (ndim < 5) {
    for(size_t i=0;i<ndim;++i){
      int abc = (14*(values.size()-i-1));
      values[i] = encoded_shape >> abc;
      encoded_shape &= (full >> (64 - abc));
    }
  } else if (ndim == 5) {
    int abc = (16*3+4);
    values[0] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (16*2+4);
    values[1] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (16*1+4);
    values[2] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (16*0+4);
    values[3] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (0);
    values[4] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));
  }else if(ndim == 6){
    int abc = (16*2+8);
    values[0] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (16*1+8);
    values[1] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (8*1+8);
    values[2] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (8);
    values[3] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (4);
    values[4] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));

    abc = (0);
    values[5] = encoded_shape >> abc;
    encoded_shape &= (full >> (64 - abc));
  }else{
    ICHECK(false) << "error layout decode failed";
  }
  // for layout IOHW1i4o
  return values;
}
template <class T>
inline int64_t encode_shape_fold(std::vector<T> values) {
  CHECK(values.size() > 0 && values.size() < 7) << " can only encode 1 to 6 dim shape";
  int64_t encoded_shape = (values.size()) << (56);
    
    //0---63bit---  w is used for winograd w*h
    //5 dim
    //|---bytes---|-shape dim-|----N----|----C------|----H----|----W-----|-----C-----|
    //|-----4-----|-----3-----|----4----|-----16----|---16----|----16----|-----4----|
    
    //6 dim
    //|---bytes---|-shape dim-|---I-----|----O------|----H----|----W-----|------1----|-----C-----|
    //|-----4-----|-----3-----|---16----|-----16----|----8----|----8-----|------4----|------4----|

    //1-4 dim
    //|---bytes---|-shape dim-|---I-----|----O------|----H----|----W-----|
    //|-----4-----|-----3-----|---14----|-----14----|----14---|----14----|
  if (values.size() <= 2) {
    for(size_t i=0;i<values.size();++i){
      int64_t va_ = std::abs(values[i]);
      ICHECK_LE(va_, 8192*8192-1) << " elments is bigger than 2^28 size=" <<values.size() << " " << values[0] <<" "<< values[1];
      encoded_shape |= va_ << (28*(values.size()-i-1));
    }
  }
  else if (values.size() <= 4) {
    for(size_t i=0;i<values.size();++i){
      int64_t va_ = std::abs(values[i]);
      ICHECK_LE(va_, 8191) << " elments is bigger than 2^14 size=" <<values.size() << " " << values[0] <<" "<< values[1];
      encoded_shape |= va_ << (14*(values.size()-i-1));
    }
  }
    // for layout IOHW1i4o
  else if (values.size() == 5) {
    int64_t va_ = std::abs(values[0]);
    ICHECK_LE(va_, 15);
    encoded_shape |= (va_ << (16*3+4));

    va_ = std::abs(values[1]);
    ICHECK_LE(va_, 65535);
    encoded_shape |= (va_ << (16*2+4));

    va_ = std::abs(values[2]);
    ICHECK_LE(va_, 65535);
    encoded_shape |= (va_ << (16*1+4));

    va_ = std::abs(values[3]);
    ICHECK_LE(va_, 65535);
    encoded_shape |= (va_ << (16*0+4));

    va_ = std::abs(values[4]);
    ICHECK_LE(va_, 15);
    encoded_shape |= va_;
  }else if (values.size() == 6){
    int64_t va_ = std::abs(values[0]);
    ICHECK_LE(va_, 65535);
    encoded_shape |= (va_ << (16*2+8));

    va_ = std::abs(values[1]);
    ICHECK_LE(va_, 65535);
    encoded_shape |= (va_ << (8*2+8));

    va_ = std::abs(values[2]);
    ICHECK_LE(va_, 256);
    encoded_shape |= (va_ << (8*1+8));

    va_ = std::abs(values[3]);
    ICHECK_LE(va_, 256);
    encoded_shape |= (va_ << (8));

    va_ = std::abs(values[4]);
    ICHECK_LE(va_, 15);
    encoded_shape |= va_ << 4;

    va_ = std::abs(values[5]);
    ICHECK_LE(va_, 15);
    encoded_shape |= va_;
  } else{
    ICHECK(false) << " doesnot surpport this layout";
  }

  ICHECK_LT(encoded_shape, std::numeric_limits<int64_t>::max());
  ICHECK_GT(encoded_shape, 0);
  return encoded_shape;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_REGISTRY_H_
