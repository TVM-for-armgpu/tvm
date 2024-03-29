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
 * \file int_set.cc
 * \brief The integer set functions
 */
#include <tvm/arith/int_set.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "interval_set.h"
#include "pattern_match.h"

namespace tvm {
namespace arith {

using tir::is_one;
using tir::is_zero;
using tir::make_const;
using tir::make_zero;

PrimExpr SymbolicLimits::pos_inf_ = Var("pos_inf", DataType::Handle());
PrimExpr SymbolicLimits::neg_inf_ = Var("neg_inf", DataType::Handle());

IntervalSet::IntervalSet(PrimExpr min_value, PrimExpr max_value) {
  auto node = make_object<IntervalSetNode>();
  node->min_value = std::move(min_value);
  node->max_value = std::move(max_value);
  data_ = std::move(node);
}

IntervalSet MakeIntervalSet(PrimExpr min_value, PrimExpr max_value) {
  return IntervalSet(min_value, max_value);
}

TVM_REGISTER_GLOBAL("arith.IntervalSet").set_body_typed(MakeIntervalSet);

IntervalSet Intersect(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  PrimExpr max_value = min(a->max_value, b->max_value);
  PrimExpr min_value = max(a->min_value, b->min_value);
  if ((max_value.dtype().is_int() || max_value.dtype().is_uint()) &&
      (min_value.dtype().is_int() || min_value.dtype().is_uint()) &&
      analyzer->CanProveGreaterEqual(min_value - max_value, 1)) {
    return IntervalSet::Empty();
  } else {
    return IntervalSet(min_value, max_value);
  }
}

IntervalSet Union(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  PrimExpr max_value = max(a->max_value, b->max_value);
  PrimExpr min_value = min(a->min_value, b->min_value);
  return IntervalSet(min_value, max_value);
}

// type traits
template <typename OP>
struct is_logical_op {
  static const bool value = false;
};

#define TVM_DECLARE_LOGICAL_OP(OP)  \
  template <>                       \
  struct is_logical_op<tir::OP> {   \
    static const bool value = true; \
  };

TVM_DECLARE_LOGICAL_OP(And);
TVM_DECLARE_LOGICAL_OP(Or);
TVM_DECLARE_LOGICAL_OP(EQ);
TVM_DECLARE_LOGICAL_OP(NE);
TVM_DECLARE_LOGICAL_OP(GE);
TVM_DECLARE_LOGICAL_OP(GT);
TVM_DECLARE_LOGICAL_OP(LE);
TVM_DECLARE_LOGICAL_OP(LT);
TVM_DECLARE_LOGICAL_OP(Not);

/*!
 * \brief Combine two interval set under arithmetic operations.
 * \note this can possibly relax the set.
 */
template <typename Op>
inline IntervalSet Combine(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    PrimExpr res = TryConstFold<Op>(a->min_value, b->min_value);
    if (!res.defined()) res = Op(a->min_value, b->min_value);
    return IntervalSet::SinglePoint(res);
  }
  if (is_logical_op<Op>::value) {
    return IntervalSet(make_const(a->min_value.dtype(), 0), make_const(a->min_value.dtype(), 1));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (a->IsEverything()) return a;
  if (b->IsEverything()) return b;
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::Add>(Analyzer* analyer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value + b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  PrimExpr min_value =
      a->HasLowerBound() && b->HasLowerBound() ? a->min_value + b->min_value : neg_inf();
  PrimExpr max_value =
      a->HasUpperBound() && b->HasUpperBound() ? a->max_value + b->max_value : pos_inf();
  return IntervalSet(min_value, max_value);
}

template <>
inline IntervalSet Combine<tir::Sub>(Analyzer* analyer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value - b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  PrimExpr min_value =
      a->HasLowerBound() && b->HasUpperBound() ? a->min_value - b->max_value : neg_inf();
  PrimExpr max_value =
      a->HasUpperBound() && b->HasLowerBound() ? a->max_value - b->min_value : pos_inf();
  return IntervalSet(min_value, max_value);
}

template <>
inline IntervalSet Combine<tir::Mul>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value * b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (a->IsSinglePoint()) {
    std::swap(a, b);
  }
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) return b;
    if (is_one(b->min_value)) return a;
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      PrimExpr min_value = a->HasLowerBound() ? a->min_value * b->min_value : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? a->max_value * b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? a->max_value * b->min_value : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? a->min_value * b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::Select;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = a->min_value * b->min_value;
      PrimExpr e2 = a->max_value * b->min_value;
      return IntervalSet(Select(sign, e1, e2), Select(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mul";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::Div>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(a->min_value / b->min_value);
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) {
      LOG(FATAL) << "Divide by zero in CombineInterval Div";
    }
    if (is_one(b->min_value)) return a;
    // no relaxation is needed in here due to set is inclusive
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      PrimExpr min_value = a->HasLowerBound() ? a->min_value / b->min_value : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? a->max_value / b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? a->max_value / b->min_value : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? a->min_value / b->min_value : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::Select;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = a->min_value / b->min_value;
      PrimExpr e2 = a->max_value / b->min_value;
      return IntervalSet(Select(sign, e1, e2), Select(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::Mod>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(truncmod(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;

  if (b->IsSinglePoint()) {
    const PrimExpr& divisor = b->min_value;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    // We need to add more bound constraints throughout the code.
    // The logic below assumes a is non-negative, which usually
    // is the case of our application.
    // TODO(tqchen): add bound constraints for a.
    if (analyzer->CanProveGreaterEqual(divisor, 0)) {
      return IntervalSet(make_zero(divisor.dtype()), divisor - 1);
    } else {
      PrimExpr bound = abs(divisor) - 1;
      return IntervalSet(-bound, bound);
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::FloorDiv>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(floordiv(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  if (b->IsSinglePoint()) {
    if (is_zero(b->min_value)) {
      LOG(FATAL) << "Divide by zero in CombineInterval Div";
    }
    if (is_one(b->min_value)) return a;
    // no relaxation is needed in here due to set is inclusive
    if (analyzer->CanProveGreaterEqual(b->min_value, 0)) {
      PrimExpr min_value = a->HasLowerBound() ? floordiv(a->min_value, b->min_value) : neg_inf();
      PrimExpr max_value = a->HasUpperBound() ? floordiv(a->max_value, b->min_value) : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (analyzer->CanProveGreaterEqual(-b->min_value, 1)) {
      PrimExpr min_value = a->HasUpperBound() ? floordiv(a->max_value, b->min_value) : neg_inf();
      PrimExpr max_value = a->HasLowerBound() ? floordiv(a->min_value, b->min_value) : pos_inf();
      return IntervalSet(min_value, max_value);
    } else if (a->HasUpperBound() && a->HasLowerBound()) {
      using tir::Select;
      PrimExpr sign = b->min_value >= make_zero(b->min_value.dtype().element_of());
      PrimExpr e1 = floordiv(a->min_value, b->min_value);
      PrimExpr e2 = floordiv(a->max_value, b->min_value);
      return IntervalSet(Select(sign, e1, e2), Select(sign, e2, e1));
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Div";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::FloorMod>(Analyzer* analyzer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(floormod(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;

  if (b->IsSinglePoint()) {
    const PrimExpr& divisor = b->min_value;
    if (is_zero(divisor)) {
      LOG(FATAL) << "Modular by zero in CombineInterval Mod";
    }
    if (analyzer->CanProveGreaterEqual(divisor, 0)) {
      if (divisor.as<tir::IntImmNode>()) {
        // a mod b = a - (a / b) * b if a_max / b == a_min / b
        auto qmax = a->HasUpperBound() ? floordiv(a->max_value, divisor) : pos_inf();
        auto qmin = a->HasLowerBound() ? floordiv(a->min_value, divisor) : neg_inf();
        if (analyzer->CanProve(qmax == qmin)) {
          auto tmax = a->max_value - divisor * qmin;
          auto tmin = a->min_value - divisor * qmin;
          return IntervalSet(tmin, tmax);
        }
      }
      return IntervalSet(make_zero(divisor.dtype()), divisor - 1);
    } else {
      PrimExpr bound = abs(divisor) - 1;
      return IntervalSet(-bound, bound);
    }
  }
  DLOG(WARNING) << "Return Everything in CombineInterval Mod";
  return IntervalSet::Everything();
}

template <>
inline IntervalSet Combine<tir::Max>(Analyzer* analzyer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(max(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  return IntervalSet(max(a->min_value, b->min_value), max(a->max_value, b->max_value));
}

template <>
inline IntervalSet Combine<tir::Min>(Analyzer* analzyer, IntervalSet a, IntervalSet b) {
  if (a->IsSinglePoint() && b->IsSinglePoint()) {
    return IntervalSet::SinglePoint(min(a->min_value, b->min_value));
  }
  if (a->IsEmpty()) return a;
  if (b->IsEmpty()) return b;
  return IntervalSet(min(a->min_value, b->min_value), min(a->max_value, b->max_value));
}

// internal helper function to get an interval set
IntervalSet ToIntervalSet(IntSet set) {
  if (auto* node = set.as<IntervalSetNode>()) {
    return GetRef<IntervalSet>(node);
  }
  DLOG(INFO) << "cannot resolve int set " << set;
  return IntervalSet::Everything();
}

using namespace tir;

// Simplified version of int set evaluator that operates on IntervalSet
// We might use better set analysis in the future to replace the intervalset.
class IntervalSetEvaluator : public ExprFunctor<IntervalSet(const PrimExpr&)> {
 public:
  IntervalSetEvaluator(Analyzer* analyzer, const Map<Var, IntSet>& dom_map, bool eval_vec = false)
      : analyzer_(analyzer), dom_map_(dom_map), eval_vec_(eval_vec) {}

  IntervalSet Eval(const PrimExpr& val) { return this->VisitExpr(val); }
  // evaluate and relax the set
  IntervalSet Eval(IntervalSet val) {
    // avoid recursive indefinite recursive expansion.
    if (static_cast<size_t>(recur_depth_) >= dom_map_.size()) return val;
    ++recur_depth_;
    IntervalSet min_set = this->Eval(val->min_value);
    IntervalSet max_set = this->Eval(val->max_value);
    --recur_depth_;
    return IntervalSet(min_set->min_value, max_set->max_value);
  }

  IntervalSet VisitExpr_(const IntImmNode* op) final {
    return IntervalSet::SinglePoint(GetRef<PrimExpr>(op));
  }

  IntervalSet VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = dom_map_.find(var);
    if (it != dom_map_.end()) {
      IntervalSet res = ToIntervalSet((*it).second);
      if (res->min_value.same_as(var) && res->max_value.same_as(var)) {
        return res;
      }
      // recursively evaluate mapped result
      // in case the domain contains variables to be relaxed.
      return Eval(res);
    } else {
      return IntervalSet::SinglePoint(var);
    }
  }

  IntervalSet VisitExpr_(const AddNode* op) final { return VisitBinaryExpr_<Add>(op); }

  IntervalSet VisitExpr_(const SubNode* op) final { return VisitBinaryExpr_<Sub>(op); }

  IntervalSet VisitExpr_(const MulNode* op) final { return VisitBinaryExpr_<Mul>(op); }

  IntervalSet VisitExpr_(const DivNode* op) final { return VisitBinaryExpr_<Div>(op); }

  IntervalSet VisitExpr_(const ModNode* op) final { return VisitBinaryExpr_<Mod>(op); }

  IntervalSet VisitExpr_(const FloorDivNode* op) final { return VisitBinaryExpr_<FloorDiv>(op); }

  IntervalSet VisitExpr_(const FloorModNode* op) final { return VisitBinaryExpr_<FloorMod>(op); }

  IntervalSet VisitExpr_(const MinNode* op) final { return VisitBinaryExpr_<Min>(op); }

  IntervalSet VisitExpr_(const MaxNode* op) final { return VisitBinaryExpr_<Max>(op); }

  IntervalSet VisitExpr_(const EQNode* op) final { return VisitBinaryExpr_<EQ>(op); }

  IntervalSet VisitExpr_(const NENode* op) final { return VisitBinaryExpr_<NE>(op); }

  IntervalSet VisitExpr_(const LTNode* op) final { return VisitBinaryExpr_<LT>(op); }

  IntervalSet VisitExpr_(const LENode* op) final { return VisitBinaryExpr_<LE>(op); }

  IntervalSet VisitExpr_(const GTNode* op) final { return VisitBinaryExpr_<GT>(op); }

  IntervalSet VisitExpr_(const GENode* op) final { return VisitBinaryExpr_<GE>(op); }

  IntervalSet VisitExpr_(const AndNode* op) final { return VisitBinaryExpr_<And>(op); }

  IntervalSet VisitExpr_(const OrNode* op) final { return VisitBinaryExpr_<Or>(op); }

  IntervalSet VisitExpr_(const RampNode* op) final {
    ICHECK(eval_vec_);
    IntervalSet base = Eval(op->base);
    PVar<IntImm> stride;
    if (stride.Match(op->stride)) {
      DataType t = op->base.dtype();
      int64_t vstride = stride.Eval()->value;
      if (vstride > 0) {
        return Combine<Add>(analyzer_, base,
                            IntervalSet(make_zero(t), make_const(t, vstride * op->lanes - 1)));
      } else {
        return Combine<Add>(analyzer_, base,
                            IntervalSet(make_const(t, vstride * op->lanes + 1), make_zero(t)));
      }
    }
    DLOG(WARNING) << "cannot evaluate set on expression " << GetRef<PrimExpr>(op);
    return IntervalSet::Everything();
  }

  IntervalSet VisitExpr_(const BroadcastNode* op) final {
    ICHECK(eval_vec_);
    return VisitExpr(op->value);
  }

  IntervalSet VisitExpr_(const SelectNode* op) final {
    IntervalSet true_set = this->Eval(op->true_value);
    IntervalSet false_set = this->Eval(op->false_value);
    return Union(analyzer_, false_set, true_set);
  }

  IntervalSet VisitExpr_(const CastNode* op) final {
    IntervalSet value_set = this->Eval(op->value);
    PrimExpr min_value =
        value_set->HasLowerBound() ? cast(op->dtype, value_set->min_value) : neg_inf();
    PrimExpr max_value =
        value_set->HasUpperBound() ? cast(op->dtype, value_set->max_value) : pos_inf();
    return IntervalSet(min_value, max_value);
  }

  IntervalSet VisitExprDefault_(const Object* op) final {
    DLOG(WARNING) << "cannot evaluate set type " << op->GetTypeKey();
    return IntervalSet::Everything();
  }

 private:
  // whether set is exactly single point that equals value.
  bool MatchPoint(const IntervalSet& set, const PrimExpr& value) const {
    return set->min_value.same_as(value) && set->max_value.same_as(value);
  }

  template <typename TOp, typename T>
  inline IntervalSet VisitBinaryExpr_(const T* op) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    IntervalSet a = this->Eval(op->a);
    IntervalSet b = this->Eval(op->b);
    if (MatchPoint(a, op->a) && MatchPoint(b, op->b)) {
      return IntervalSet::SinglePoint(GetRef<PrimExpr>(op));
    }
    return Combine<TOp>(analyzer_, a, b);
  }

  // recursive depth
  int recur_depth_{0};
  // analyzer
  Analyzer* analyzer_;
  const Map<Var, IntSet>& dom_map_;
  bool eval_vec_{false};
};

class IntSetAnalyzer::Impl {
 public:
  explicit Impl(Analyzer* analyzer) : analyzer_(analyzer) {}

  IntSet Eval(const PrimExpr& expr, const Map<Var, IntSet>& dom_map) const {
    return IntervalSetEvaluator(analyzer_, dom_map).Eval(expr);
  }

 private:
  Analyzer* analyzer_;
};

IntSetAnalyzer::IntSetAnalyzer(Analyzer* parent) : impl_(new Impl(parent)) {}

IntSetAnalyzer::~IntSetAnalyzer() { delete impl_; }

IntSet IntSetAnalyzer::operator()(const PrimExpr& expr, const Map<Var, IntSet>& dom_map) {
  return impl_->Eval(expr, dom_map);
}

// Quickly adapt to IntSet interface
// TODO(tqchen): revisit IntSet interface as well.
Range IntSet::CoverRange(Range max_range) const {
  IntSet temp;
  Analyzer analyzer;
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  ICHECK(s_int != nullptr);
  if (s_int->HasUpperBound() && s_int->HasLowerBound()) {
    return Range::FromMinExtent(s_int->min_value,
                                analyzer.Simplify(s_int->max_value + 1 - s_int->min_value));
  }
  return max_range;
}

PrimExpr IntSet::min() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  ICHECK(s_int);
  return s_int->min_value;
}

PrimExpr IntSet::max() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  ICHECK(s_int);
  return s_int->max_value;
}

bool IntSet::IsNothing() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsEmpty());
}

bool IntSet::IsEverything() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsEverything());
}

bool IntSet::IsSinglePoint() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && s_int->IsSinglePoint());
}

bool IntSet::CanProvePositive() const {
  Analyzer analyzer;
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && is_positive_const(analyzer.Simplify(s_int->min_value)));
}

bool IntSet::CanProveNegative() const {
  Analyzer analyzer;
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  return (s_int && is_negative_const(analyzer.Simplify(s_int->max_value)));
}

bool IntSet::CanProveNonPositive() const {
  Analyzer analyzer;
  if (const auto* s_int = (*this).as<IntervalSetNode>()) {
    auto max = analyzer.Simplify(s_int->max_value);
    return is_zero(max) || is_negative_const(max);
  }
  return false;
}

bool IntSet::CanProveNonNegative() const {
  Analyzer analyzer;
  if (const IntervalSetNode* s_int = (*this).as<IntervalSetNode>()) {
    auto min = analyzer.Simplify(s_int->min_value);
    return is_zero(min) || is_positive_const(min);
  }
  return false;
}

SignType IntSet::GetSignType() const {
  if (CanProvePositive()) {
    return kPositive;
  } else if (CanProveNegative()) {
    return kNegative;
  } else if (IsSinglePoint() && is_zero(PointValue())) {
    return kZero;
  } else {
    return kUnknown;
  }
}
PrimExpr IntSet::PointValue() const {
  const IntervalSetNode* s_int = (*this).as<IntervalSetNode>();
  ICHECK(s_int && s_int->IsSinglePoint());
  return s_int->min_value;
}

IntSet IntSet::Nothing() { return IntervalSet::Empty(); }

IntSet IntSet::Everything() { return IntervalSet::Everything(); }

IntSet IntSet::SinglePoint(PrimExpr x) { return IntervalSet::SinglePoint(x); }

IntSet IntSet::Interval(PrimExpr min, PrimExpr max) {
  if (min.same_as(max)) {
    return IntSet::SinglePoint(min);
  }
  return IntervalSet(min, max);
}


struct EliminableSet;
EliminableSet::EliminableSet(PrimExpr expr_, PrimExpr max_value_) {
  auto n = make_object<EliminableSetNode>();
  n->max_value = std::move(max_value_);
  n->expr = std::move(expr_);
  data_ = std::move(n);
}
// Simplified version of int set evaluator that operates on IntervalSet
// We might use better set analysis in the future to replace the intervalset.
class InequationEvaluator : public ExprFunctor<EliminableSet(const PrimExpr&)> {
 public:
  InequationEvaluator(Analyzer* analyzer, const Map<Var, IntSet>& dom_map)
      : analyzer_(analyzer), dom_map_(dom_map) {}
  typedef EliminableSet ExprEntry;
  ExprEntry Eval(const PrimExpr& val) { return this->VisitExpr(val); }
  // return itself if cant be eliminated else return IntImm(DataType::Int(32), 0)
  ExprEntry VisitExpr_(const IntImmNode* op) final {
    ExprEntry entry(IntImm(DataType::Int(32), 0), GetRef<IntImm>(op));
    return entry;
  }

  ExprEntry VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = dom_map_.find(var);
    if (it != dom_map_.end()) {
      IntervalSet res = ToIntervalSet((*it).second);
      ExprEntry entry(IntImm(DataType::Int(32), 0), std::move(res->max_value));
      return entry;
    }
    return ExprEntry(std::move(var), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const AddNode* op) final {
    ExprEntry a = this->Eval(op->a);
    ExprEntry b = this->Eval(op->b);
    // if and only if all of a,b can be eliminated, this op is also eleminated
    bool elim_a = CanEliminate(a, op->a);
    bool elim_b = CanEliminate(b, op->b);
    if (elim_a) {
      if (elim_b) {
        return ExprEntry(IntImm(DataType::Int(32), 0), Add(a->max_value, b->max_value));
      } else {
        return ExprEntry(op->b, a->max_value);
      }
    } else {
      if (elim_b) {
        return ExprEntry(op->a, b->max_value);
      } else {
        return ExprEntry(GetRef<Add>(op), IntImm(DataType::Int(32), 0));
      }
    }
  }

  ExprEntry VisitExpr_(const SubNode* op) final {
    if (!(op->b.as<IntImmNode>())) {
      return ExprEntry(GetRef<Sub>(op), IntImm(DataType::Int(32), 0));
    }
    ExprEntry a = this->Eval(op->a);
    ExprEntry b = this->Eval(op->b);
    // if and only if all of a,b can be eliminated, this op is also eleminated
    bool elim_a = CanEliminate(a, op->a);
    bool elim_b = CanEliminate(b, op->b);
    if (elim_a) {
      if (elim_b) {
        IntervalSet rg = IntervalSetEvaluator(analyzer_, dom_map_).Eval(GetRef<Sub>(op));
        return ExprEntry(IntImm(DataType::Int(32), 0), rg->max_value);
      } else {
        return ExprEntry(-1 * op->b, a->max_value);
      }
    } else {
      if (elim_b) {
        return ExprEntry(op->a, -1 * b->max_value);
      } else {
        return ExprEntry(GetRef<Sub>(op), IntImm(DataType::Int(32), 0));
      }
    }
  }

  ExprEntry VisitExpr_(const MulNode* op) final { return VisitBinaryExpr_<Mul>(op); }

  ExprEntry VisitExpr_(const DivNode* op) final { return VisitBinaryExpr_<Div>(op); }

  ExprEntry VisitExpr_(const ModNode* op) final { return VisitBinaryExpr_<Mod>(op); }

  ExprEntry VisitExpr_(const FloorDivNode* op) final { return VisitBinaryExpr_<FloorDiv>(op); }

  ExprEntry VisitExpr_(const FloorModNode* op) final { return VisitBinaryExpr_<FloorMod>(op); }

  ExprEntry VisitExpr_(const MinNode* op) final { return VisitBinaryExpr_<Min>(op); }

  ExprEntry VisitExpr_(const MaxNode* op) final { return VisitBinaryExpr_<Max>(op); }

  ExprEntry VisitExpr_(const EQNode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const NENode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const LTNode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const LENode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const GTNode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const GENode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const AndNode* op) final { return VisitBinaryExpr_<And>(op); }

  ExprEntry VisitExpr_(const OrNode* op) final { return VisitBinaryExpr_<Or>(op); }

  ExprEntry VisitExpr_(const RampNode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const BroadcastNode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const SelectNode* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExpr_(const CastNode* op) final {
    return ExprEntry(GetRef<Cast>(op), IntImm(DataType::Int(32), 0));
  }

  ExprEntry VisitExprDefault_(const Object* op) final {
    ICHECK(false) << "can't be processed";
    return ExprEntry(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), 0));
  }

 private:
  // whether set is exactly single point that equals value.
  bool CanEliminate(const ExprEntry& set, const PrimExpr& value) const {
    return (!(set->expr.same_as(value))) &&
           (!(set->max_value.same_as(IntImm(DataType::Int(32), 0))));
  }

  template <typename TOp, typename T>
  inline ExprEntry VisitBinaryExpr_(const T* op) {
    static_assert(std::is_same<typename TOp::ContainerType, T>::value, "constraint");
    ExprEntry a = this->Eval(op->a);
    ExprEntry b = this->Eval(op->b);
    // if and only if all of a,b can be eliminated, this op is also eleminated
    if (CanEliminate(a, op->a) && CanEliminate(b, op->b)) {
      IntervalSet rg = IntervalSetEvaluator(analyzer_, dom_map_).Eval(GetRef<TOp>(op));
      ExprEntry ret(IntImm(DataType::Int(32), 0), rg->max_value);
      return ret;
    }
    return ExprEntry(std::move(GetRef<TOp>(op)), IntImm(DataType::Int(32), 0));
  }

  // recursive depth
  int recur_depth_{0};
  // analyzer
  Analyzer* analyzer_;
  const Map<Var, IntSet>& dom_map_;
  bool eval_vec_{false};
};

class InequationAnalyzer::IneqImpl {
 public:
  explicit IneqImpl(Analyzer* analyzer) : analyzer_(analyzer) {}

  PrimExpr Eval(const PrimExpr& expr, const Map<Var, IntSet>& dom_map) const {
    if (auto* ptr = expr.as<LTNode>()) {
      if (!(ptr->b.as<IntImmNode>())) {
        return expr;
      }
      EliminableSet entry = InequationEvaluator(analyzer_, dom_map).Eval(ptr->a);
      return LT(entry->expr, analyzer_->rewrite_simplify(ptr->b - entry->max_value));
    }
    return expr;
  }

 private:
  Analyzer* analyzer_;
};

InequationAnalyzer::InequationAnalyzer(Analyzer* parent) : impl_(new IneqImpl(parent)) {}

InequationAnalyzer::~InequationAnalyzer() { delete impl_; }

PrimExpr InequationAnalyzer::operator()(const PrimExpr& expr, const Map<Var, IntSet>& dom_map) {
  return impl_->Eval(expr, dom_map);
}
TVM_REGISTER_NODE_TYPE(EliminableSetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EliminableSetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const EliminableSetNode*>(node.get());
      p->stream << "EliminableSet"
                << "[" << op->expr << ", " << op->max_value << ']';
    });


// Range related code
inline bool ProveEqual(Analyzer* analyzer, PrimExpr lhs, PrimExpr rhs) {
  return is_zero(analyzer->Simplify(lhs - rhs));
}

IntSet IntSet::FromRange(Range r) {
  // must make sure it can be matched back by MatchRange.
  if (is_one(r->extent)) {
    return IntSet::SinglePoint(r->min);
  }
  return IntervalSet(r->min, r->extent + r->min - 1);
}

bool IntSet::MatchRange(const Range& b) const {
  const IntSet& a = *this;
  const IntervalSetNode* a_int = a.as<IntervalSetNode>();
  if (!a_int) return false;
  if (!a_int->HasUpperBound() || !a_int->HasLowerBound()) return false;
  Analyzer ana;
  return ProveEqual(&ana, a_int->min_value, b->min) &&
         ProveEqual(&ana, a_int->max_value, b->extent + b->min - 1);
}

IntSet Union(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::Nothing();
  if (sets.size() == 1) return sets[0];
  Analyzer ana;
  IntervalSet x = ToIntervalSet(sets[0]);
  for (size_t i = 1; i < sets.size(); ++i) {
    x = Union(&ana, x, ToIntervalSet(sets[i]));
  }
  return IntervalSet(ana.Simplify(x->min_value), ana.Simplify(x->max_value));
}

IntSet Intersect(const Array<IntSet>& sets) {
  if (sets.size() == 0) return IntSet::Nothing();
  if (sets.size() == 1) return sets[0];
  Analyzer ana;
  IntervalSet x = ToIntervalSet(sets[0]);
  for (size_t i = 1; i < sets.size(); ++i) {
    x = Intersect(&ana, x, ToIntervalSet(sets[i]));
  }
  return IntervalSet(ana.Simplify(x->min_value), ana.Simplify(x->max_value));
}

Map<Var, IntSet> ConvertDomMap(const Map<IterVar, IntSet>& dom_map) {
  Map<Var, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap.Set(kv.first->var, kv.second);
  }
  return dmap;
}

Map<Var, IntSet> ConvertDomMap(const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Map<Var, IntSet> dmap;
  for (auto kv : dom_map) {
    dmap.Set(GetRef<Var>(kv.first), kv.second);
  }
  return dmap;
}

IntSet EvalSet(PrimExpr e, const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  return IntervalSetEvaluator(&ana, dom_map, false).Eval(e);
}

IntSet IntSet::Vector(PrimExpr x) {
  Analyzer ana;
  Map<Var, IntSet> dmap;
  return IntervalSetEvaluator(&ana, dmap, true).Eval(x);
}

IntSet EvalSet(PrimExpr e, const Map<IterVar, IntSet>& dom_map) {
  return EvalSet(e, ConvertDomMap(dom_map));
}

IntSet EvalSet(PrimExpr e, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  return EvalSet(e, ConvertDomMap(dom_map));
}

IntSet EvalSet(Range r, const Map<Var, IntSet>& dom_map) {
  Analyzer ana;
  IntervalSetEvaluator m(&ana, dom_map);
  // Simplifying first can give tighter bounds if r->min and r->extent share variables
  PrimExpr sum = r->min + r->extent - 1;
  auto res = m.Eval(IntervalSet(r->min, ana.Simplify(sum)));
  return std::move(res);
}

IntSet EvalSet(Range r, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  return EvalSet(r, ConvertDomMap(dom_map));
}

IntSet EvalSet(IntSet s, const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Analyzer ana;
  auto dmap = ConvertDomMap(dom_map);
  IntervalSetEvaluator m(&ana, dmap);
  const IntervalSetNode* s_int = s.as<IntervalSetNode>();
  PrimExpr vmax = s_int->HasUpperBound() ? m.Eval(s_int->max_value).max() : s_int->max_value;
  PrimExpr vmin = s_int->HasLowerBound() ? m.Eval(s_int->min_value).min() : s_int->min_value;
  return IntervalSet(vmin, vmax);
}

class SubExprIntervalSetEvaluator : public IntervalSetEvaluator {
 public:
  explicit SubExprIntervalSetEvaluator(Analyzer* analyzer, const Map<Var, IntSet>& dom_map)
      : IntervalSetEvaluator(analyzer, dom_map) {}

  IntervalSet VisitExpr(const PrimExpr& n) final {
    IntervalSet ret = IntervalSetEvaluator::VisitExpr(n);
    expr_map[n] = ret;
    return ret;
  }

  ExprIntSetMap expr_map;
};

ExprIntSetMap EvalSetForEachSubExpr(PrimExpr e,
                                    const std::unordered_map<const VarNode*, IntSet>& dom_map) {
  Analyzer ana;
  auto dmap = ConvertDomMap(dom_map);
  SubExprIntervalSetEvaluator m(&ana, dmap);
  m.Eval(e);
  return m.expr_map;
}

IntSet EvalSet(Range r, const Map<IterVar, IntSet>& dom_map) {
  return EvalSet(r, ConvertDomMap(dom_map));
}

TVM_REGISTER_NODE_TYPE(IntervalSetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IntervalSetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IntervalSetNode*>(node.get());
      p->stream << "IntervalSet"
                << "[" << op->min_value << ", " << op->max_value << ']';
    });

TVM_REGISTER_GLOBAL("arith.intset_single_point").set_body_typed(IntSet::SinglePoint);

TVM_REGISTER_GLOBAL("arith.intset_vector").set_body_typed(IntSet::Vector);

TVM_REGISTER_GLOBAL("arith.intset_interval").set_body_typed(IntSet::Interval);

TVM_REGISTER_GLOBAL("arith.IntervalSetGetMin").set_body_method(&IntSet::min);

TVM_REGISTER_GLOBAL("arith.IntervalSetGetMax").set_body_method(&IntSet::max);

TVM_REGISTER_GLOBAL("arith.IntSetIsNothing").set_body_method(&IntSet::IsNothing);

TVM_REGISTER_GLOBAL("arith.IntSetIsEverything").set_body_method(&IntSet::IsEverything);

}  // namespace arith
}  // namespace tvm
