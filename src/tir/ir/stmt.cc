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
 * \file tvm/tir/stmt.cc
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace tir {

// LetStmt
LetStmt::LetStmt(Var var, PrimExpr value, Stmt body, Span span) {
  ICHECK(value.defined());
  ICHECK(body.defined());
  ICHECK_EQ(value.dtype(), var.dtype());

  ObjectPtr<LetStmtNode> node = make_object<LetStmtNode>();
  node->var = std::move(var);
  node->value = std::move(value);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.LetStmt")
    .set_body_typed([](Var var, PrimExpr value, Stmt body, Span span) {
      return LetStmt(var, value, body, span);
    });

TVM_REGISTER_NODE_TYPE(LetStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LetStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const LetStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "let " << op->var << " = ";
      p->Print(op->value);
      p->stream << '\n';
      p->Print(op->body);
    });

// AttrStmt
AttrStmt::AttrStmt(ObjectRef node, String attr_key, PrimExpr value, Stmt body, Span span) {
  auto n = make_object<AttrStmtNode>();
  n->node = node;
  n->attr_key = std::move(attr_key);
  n->value = std::move(value);
  n->body = std::move(body);
  n->span = std::move(span);
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tir.AttrStmt")
    .set_body_typed([](ObjectRef node, String attr_key, PrimExpr value, Stmt body, Span span) {
      return AttrStmt(node, attr_key, value, body, span);
    });

TVM_REGISTER_NODE_TYPE(AttrStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AttrStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AttrStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "// attr [";
      p->Print(op->node);
      p->stream << "] " << op->attr_key << " = ";
      p->Print(op->value);
      p->stream << '\n';
      p->Print(op->body);
    });

// AssertStmt
AssertStmt::AssertStmt(PrimExpr condition, PrimExpr message, Stmt body, Span span) {
  ICHECK(condition.defined());
  ICHECK(message.dtype() == DataType::Int(32) || message.as<StringImmNode>())
      << "TypeError: AssertStmt message must be an int or string:" << message << "\n";

  ObjectPtr<AssertStmtNode> node = make_object<AssertStmtNode>();
  node->condition = std::move(condition);
  node->message = std::move(message);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(AssertStmtNode);

TVM_REGISTER_GLOBAL("tir.AssertStmt")
    .set_body_typed([](PrimExpr condition, ObjectRef message, Stmt body, Span span) {
      if (const auto* str = message.as<StringObj>()) {
        auto msg = StringImm(str->data);
        return AssertStmt(condition, msg, body, span);
      } else {
        return AssertStmt(condition, Downcast<PrimExpr>(message), body, span);
      }
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AssertStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AssertStmtNode*>(node.get());
      p->PrintIndent();
      p->stream << "assert(";
      p->Print(op->condition);
      p->stream << ", ";
      p->Print(op->message);
      p->stream << ")\n";
      p->Print(op->body);
    });

// For
For::For(Var loop_var, PrimExpr min, PrimExpr extent, ForKind kind, Stmt body,
         Optional<IterVar> thread_binding, Map<String, ObjectRef> annotations, Span span) {
  ICHECK(min.defined());
  ICHECK(extent.defined());
  ICHECK(min.dtype().is_scalar());
  ICHECK(extent.dtype().is_scalar());
  ICHECK(loop_var.dtype().is_scalar());
  ICHECK(body.defined());

  ObjectPtr<ForNode> node = make_object<ForNode>();
  node->loop_var = std::move(loop_var);
  node->min = std::move(min);
  node->extent = std::move(extent);
  node->kind = kind;
  node->body = std::move(body);
  node->thread_binding = std::move(thread_binding);
  node->annotations = std::move(annotations);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.For").set_body_typed(
    [](Var loop_var, PrimExpr min, PrimExpr extent, int kind, Stmt body,
       Optional<IterVar> thread_binding, Optional<Map<String, ObjectRef>> annotations, Span span) {
      return For(loop_var, min, extent, static_cast<ForKind>(kind), body, thread_binding,
                 annotations.value_or(Map<String, ObjectRef>()), span);
    });

TVM_REGISTER_NODE_TYPE(ForNode);

std::ostream& operator<<(std::ostream& out, ForKind type) {  // NOLINT(*)
  switch (type) {
    case ForKind::kSerial:
      out << "for";
      break;
    case ForKind::kParallel:
      out << "parallel";
      break;
    case ForKind::kUnrolled:
      out << "unrolled";
      break;
    case ForKind::kVectorized:
      out << "vectorized";
      break;
    case ForKind::kThreadBinding:
      out << "launch_thread";
      break;
  }
  return out;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ForNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ForNode*>(node.get());
      p->PrintIndent();
      p->stream << op->kind << " (" << op->loop_var << ", ";
      p->Print(op->min);
      p->stream << ", ";
      p->Print(op->extent);
      p->stream << ") {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

// Store
Store::Store(Var buffer_var, PrimExpr value, PrimExpr index, PrimExpr predicate, Span span) {
  ICHECK(value.defined());
  ICHECK(index.defined());
  ICHECK(predicate.defined());
  ICHECK_EQ(value.dtype().lanes(), index.dtype().lanes());
  ICHECK_EQ(value.dtype().lanes(), predicate.dtype().lanes());

  ObjectPtr<StoreNode> node = make_object<StoreNode>();
  node->buffer_var = std::move(buffer_var);
  if ((value->dtype.is_climgfloatrw()) ||
    (value->value_storage_type == DataType::kCLImgFloatW)||
    (value->value_storage_type == DataType::kCLImgFloat) ){
    if (std::string(node->buffer_var->name_hint).find(".") == std::string::npos) {
      value.mutable_dtype() = value.mutable_dtype().with_code(DataType::kCLImgFloatW);
      node->value_storage_type = DataType::kCLImgFloatW;
      value.mutable_storage_type() = DataType::kCLImgFloatW;
    }else{
      //value.mutable_dtype() = value.mutable_dtype().with_code(DataType::kFloat);
      node->value_storage_type = DataType::kCLImgFloatW;
      value.mutable_storage_type() = DataType::kCLImgFloatW;
    }
  }
  node->value = std::move(value);
  node->index = std::move(index);
  node->predicate = std::move(predicate);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Store").set_body([](TVMArgs args, TVMRetValue* ret) {
  PrimExpr value = args[1];
  if (args.size() == 3) {
    *ret = Store(args[0], value, args[2], const_true(value.dtype().lanes()), Span());
  } else if (args.size() == 4) {
    *ret = Store(args[0], value, args[2], args[3], Span());
  } else {
    *ret = Store(args[0], value, args[2], args[3], args[4]);
  }
});

TVM_REGISTER_NODE_TYPE(StoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<StoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const StoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->buffer_var << "[";
      p->Print(op->index);
      p->stream << "] = ";
      p->Print(op->value);
      if (!is_one(op->predicate)) {
        p->stream << " if ";
        p->Print(op->predicate);
      }
      p->stream << '\n';
    });

// ProducerStore
ProducerStore::ProducerStore(DataProducer producer, PrimExpr value, Array<PrimExpr> indices,
                             Span span) {
  ObjectPtr<ProducerStoreNode> node = make_object<ProducerStoreNode>();
  node->producer = std::move(producer);
  node->value = std::move(value);
  if (node->producer->GetDataType().is_climgfloat() || node->producer->GetDataType().is_climgfloatw()) {
    node->value.mutable_storage_type() = DataType::kCLImgFloatW;
  }
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerStore")
    .set_body_typed([](DataProducer producer, PrimExpr value, Array<PrimExpr> indices, Span span) {
      return ProducerStore(producer, value, indices, span);
    });

TVM_REGISTER_NODE_TYPE(ProducerStoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProducerStoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProducerStoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->producer->GetNameHint() << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) p->stream << ", ";
      }
      p->stream << "]";
      p->stream << " =";
      p->Print(op->value);
      p->stream << '\n';
    });

// Allocate
Allocate::Allocate(Var buffer_var, DataType dtype, Array<PrimExpr> extents, PrimExpr condition,
                   Stmt body, Span span) {
  CHECK(IsPointerType(buffer_var->type_annotation, dtype))
      << "The allocated data type (" << dtype
      << ") does not match the type annotation of the buffer " << buffer_var << " ("
      << buffer_var->type_annotation
      << "). The data type should be an element of the pointer type.";

  for (size_t i = 0; i < extents.size(); ++i) {
    ICHECK(extents[i].defined());
    ICHECK(extents[i].dtype().is_scalar());
  }
  ICHECK(body.defined());
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_bool());

  ObjectPtr<AllocateNode> node = make_object<AllocateNode>();
  node->buffer_var = std::move(buffer_var);
  node->dtype = dtype;
  node->extents = std::move(extents);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

int32_t AllocateNode::constant_allocation_size(const Array<PrimExpr>& extents) {
  int64_t result = 1;
  // encode all shapes into int64
  if (extents.size() == 1 && extents[0].dtype().bits() == 64) {
    std::vector<int> shapes = tvm::runtime::decode_shape_fold<int>(extents[0].as<IntImmNode>()->value);
    for (size_t i = 0; i < shapes.size(); ++i) {
      result *= shapes[i];
      if (result > std::numeric_limits<int32_t>::max()) {
        return 0;
      }
    }
    return static_cast<int32_t>(result);
  }
  for (size_t i = 0; i < extents.size(); ++i) {
    if (const IntImmNode* int_size = extents[i].as<IntImmNode>()) {
      result *= int_size->value;
      if (result > std::numeric_limits<int32_t>::max()) {
        return 0;
      }
    } else {
      return 0;
    }
  }
  return static_cast<int32_t>(result);
}

TVM_REGISTER_GLOBAL("tir.Allocate")
    .set_body_typed([](Var buffer_var, DataType type, Array<PrimExpr> extents, PrimExpr condition,
                       Stmt body, Span span) {
      return Allocate(buffer_var, type, extents, condition, body, span);
    });

TVM_REGISTER_NODE_TYPE(AllocateNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<AllocateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const AllocateNode*>(node.get());
      p->PrintIndent();
      p->stream << "allocate " << op->buffer_var << "[" << op->dtype;
      for (size_t i = 0; i < op->extents.size(); ++i) {
        p->stream << " * ";
        p->Print(op->extents[i]);
      }
      p->stream << "]";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << "\n";
      p->Print(op->body);
    });

// ProducerRealize
ProducerRealize::ProducerRealize(DataProducer producer, Region bounds, PrimExpr condition,
                                 Stmt body, Span span) {
  for (size_t i = 0; i < bounds.size(); ++i) {
    ICHECK(bounds[i]->min.defined());
    ICHECK(bounds[i]->extent.defined());
    ICHECK(bounds[i]->min.dtype().is_scalar());
    ICHECK(bounds[i]->extent.dtype().is_scalar());
  }
  ICHECK(body.defined());
  ICHECK(condition.defined());
  ICHECK(condition.dtype().is_bool());

  ObjectPtr<ProducerRealizeNode> node = make_object<ProducerRealizeNode>();
  node->producer = std::move(producer);
  node->bounds = std::move(bounds);
  node->condition = std::move(condition);
  node->body = std::move(body);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.ProducerRealize")
    .set_body_typed([](DataProducer producer, Region bounds, PrimExpr condition, Stmt body,
                       Span span) {
      return ProducerRealize(producer, bounds, condition, body, span);
    });

TVM_REGISTER_NODE_TYPE(ProducerRealizeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ProducerRealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const ProducerRealizeNode*>(node.get());
      p->PrintIndent();
      p->stream << "producer_realize " << op->producer->GetNameHint() << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

// Prefetch
Prefetch::Prefetch(Buffer buffer, Array<Range> bounds, Span span) {
  data_ = make_object<PrefetchNode>(buffer, bounds, span);
}

TVM_REGISTER_GLOBAL("tir.Prefetch")
    .set_body_typed([](Buffer buffer, Array<Range> bounds, Span span) {
      return Prefetch(buffer, bounds, span);
    });

TVM_REGISTER_NODE_TYPE(PrefetchNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrefetchNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const PrefetchNode*>(node.get());
      p->PrintIndent();
      p->stream << "prefetch " << op->buffer << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
    });

// SeqStmt
SeqStmt::SeqStmt(Array<Stmt> seq, Span span) {
  auto node = make_object<SeqStmtNode>();
  node->seq = std::move(seq);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.SeqStmt").set_body_typed([](Array<Stmt> seq, Span span) {
  return SeqStmt(std::move(seq), span);
});

TVM_REGISTER_NODE_TYPE(SeqStmtNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SeqStmtNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SeqStmtNode*>(node.get());
      for (Stmt stmt : op->seq) {
        p->Print(stmt);
      }
    });

// IfThenElse
IfThenElse::IfThenElse(PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
  ICHECK(condition.defined());
  ICHECK(then_case.defined());
  // else_case may be null.
  ObjectPtr<IfThenElseNode> node = make_object<IfThenElseNode>();
  node->condition = std::move(condition);
  node->then_case = std::move(then_case);
  node->else_case = std::move(else_case);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_NODE_TYPE(IfThenElseNode);

TVM_REGISTER_GLOBAL("tir.IfThenElse")
    .set_body_typed([](PrimExpr condition, Stmt then_case, Stmt else_case, Span span) {
      return IfThenElse(condition, then_case, else_case, span);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IfThenElseNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IfThenElseNode*>(node.get());
      p->PrintIndent();
      while (true) {
        p->stream << "if (" << op->condition << ") {\n";
        p->indent += 2;
        p->Print(op->then_case);
        p->indent -= 2;

        if (!op->else_case.defined()) {
          break;
        }

        if (const IfThenElseNode* nested_if = op->else_case.as<IfThenElseNode>()) {
          p->PrintIndent();
          p->stream << "} else ";
          op = nested_if;
        } else {
          p->PrintIndent();
          p->stream << "} else {\n";
          p->indent += 2;
          p->Print(op->else_case);
          p->indent -= 2;
          break;
        }
      }
      p->PrintIndent();
      p->stream << "}\n";
    });

// Evaluate
Evaluate::Evaluate(PrimExpr value, Span span) {
  ICHECK(value.defined());

  ObjectPtr<EvaluateNode> node = make_object<EvaluateNode>();
  node->value = std::move(value);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.Evaluate").set_body_typed([](PrimExpr value, Span span) {
  return Evaluate(value, span);
});

TVM_REGISTER_NODE_TYPE(EvaluateNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<EvaluateNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const EvaluateNode*>(node.get());
      p->PrintIndent();
      p->Print(op->value);
      p->stream << "\n";
    });

// BufferStore
BufferStore::BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices, Span span) {
  ObjectPtr<BufferStoreNode> node = make_object<BufferStoreNode>();
  node->buffer = std::move(buffer);
  node->value = std::move(value);
  if (node->buffer.get()->dtype.is_climgfloat() 
      || node->value->dtype.is_climgfloat()
      || node->value->dtype.is_climgfloatw()) {
    node->value.mutable_storage_type() = DataType::kCLImgFloatW;
  }
  node->indices = std::move(indices);
  node->span = std::move(span);
  data_ = std::move(node);
}

TVM_REGISTER_GLOBAL("tir.BufferStore")
    .set_body_typed([](Buffer buffer, PrimExpr value, Array<PrimExpr> indices, Span span) {
      return BufferStore(buffer, value, indices, span);
    });

TVM_REGISTER_NODE_TYPE(BufferStoreNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferStoreNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferStoreNode*>(node.get());
      p->PrintIndent();
      p->stream << op->buffer->name << "[";
      for (size_t i = 0; i < op->indices.size(); ++i) {
        p->Print(op->indices[i]);
        if (i < op->indices.size() - 1) p->stream << ", ";
      }
      p->stream << "]";
      p->stream << " = ";
      p->Print(op->value);
      p->stream << '\n';
    });

// BufferRealize
BufferRealize::BufferRealize(Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body,
                             Span span) {
  data_ = make_object<BufferRealizeNode>(buffer, bounds, condition, body, span);
}

TVM_REGISTER_GLOBAL("tir.BufferRealize")
    .set_body_typed([](Buffer buffer, Array<Range> bounds, PrimExpr condition, Stmt body,
                       Span span) { return BufferRealize(buffer, bounds, condition, body, span); });

TVM_REGISTER_NODE_TYPE(BufferRealizeNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BufferRealizeNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const BufferRealizeNode*>(node.get());
      p->PrintIndent();
      p->stream << "buffer_realize " << op->buffer->name << "(";
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        p->stream << "[";
        p->Print(op->bounds[i]->min);
        p->stream << ", ";
        p->Print(op->bounds[i]->extent);
        p->stream << "]";
        if (i < op->bounds.size() - 1) p->stream << ", ";
      }
      p->stream << ")";
      if (!is_one(op->condition)) {
        p->stream << " if ";
        p->Print(op->condition);
      }
      p->stream << " {\n";

      p->indent += 2;
      p->Print(op->body);
      p->indent -= 2;

      p->PrintIndent();
      p->stream << "}\n";
    });

PrimExpr TypeAnnotation(DataType dtype, Span span) {
  static auto op = Op::Get("tir.type_annotation");
  return tir::Call(dtype, op, {}, span);
}

TVM_REGISTER_OP("tir.type_annotation")
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure));

}  // namespace tir
}  // namespace tvm
