#include <iostream>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <vector>
#include <regex>
#include <assert.h>
#include <tvm/support/logging.h>
namespace tvm {
namespace tir {
namespace exprSimp {

enum TokenType {
  L_TOKEN_PUNCTUATION,
  L_TOKEN_SYMBOL,
  L_TOKEN_INTEGER,
};

struct Token {
  TokenType ty;
  std::string punctuation;
  std::string symbol;
  int integer;
};

inline bool is_valid_punctuation(const std::string& c) {
  return c == "(" || c == ")" || c == "+" || c == "*" || c == "/" || c == "%" || c == "&" ||
         c == "|" || c == "<<" || c == ">>";
}

constexpr bool is_expon_by_2(int32_t number) { return (number & number - 1) == 0; }
constexpr uint32_t get_nearsest_expo_by2(uint32_t a) {
  int n = a - 1;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

constexpr int32_t mylog(int32_t value) {
  int x = 0;
  while (value > 1) {
    value >>= 1;
    x++;
  }
  return x;
}

std::vector<Token> lex(const char* expr) {
  if (expr == nullptr) {
    return {};
  }

  std::vector<Token> rv;

  rv.push_back(Token{L_TOKEN_PUNCTUATION, "("});

  TokenType last_ty = L_TOKEN_PUNCTUATION;
  const char* beg = expr;
  const char* pos = expr;
  int32_t expr_len = strlen(expr);
  auto next_token = [pos]() { return *(pos + 1); };
  for (;;) {
    const char c = *pos;

    bool is_digit = c >= '0' && c <= '9';
    bool is_alphabet = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c == '_');
    bool is_whitespace = c == ' ' || c == '\t';
    bool is_shift = c == '<' || c == '>';

    TokenType ty;
    switch (last_ty) {
      case L_TOKEN_PUNCTUATION:
        if (is_digit) {
          ty = L_TOKEN_INTEGER;
        } else if (is_alphabet) {
          ty = L_TOKEN_SYMBOL;
        } else {
          ty = L_TOKEN_PUNCTUATION;
        }
        break;
      case L_TOKEN_INTEGER:
        if (is_digit) {
          ty = L_TOKEN_INTEGER;
        } else if (is_alphabet) {
          throw std::logic_error("numbers cannot preceed alphabets");
        } else {
          ty = L_TOKEN_PUNCTUATION;
        }
        break;
      case L_TOKEN_SYMBOL:
        if (is_digit || is_alphabet) {
          ty = L_TOKEN_SYMBOL;
        } else {
          ty = L_TOKEN_PUNCTUATION;
        }
        break;
    }

    if (beg != pos && (ty == L_TOKEN_PUNCTUATION || ty != last_ty) &&
        // Ignore the first char of shift operators.
        !(ty == L_TOKEN_PUNCTUATION && is_shift && pos == beg + 1)) {
      std::string token_lit(beg, pos);
      beg = pos;

      if (token_lit[0] != ' ' || token_lit[0] != '\t') {
        Token token{};
        token.ty = last_ty;
        switch (last_ty) {
          case L_TOKEN_PUNCTUATION:
            if (token_lit.size() > 2) {
              throw std::logic_error("punctuation must have less than two chars");
            }
            if (is_valid_punctuation(token_lit)) {
              token.punctuation = token_lit;
            } else {
              throw std::runtime_error("unrecognized punctuation");
            }
            break;
          case L_TOKEN_INTEGER:
            token.integer = std::stoi(token_lit);
            break;
          case L_TOKEN_SYMBOL:
            token.symbol = token_lit;
            break;
        }

        rv.push_back(token);
      }
    }

    if (c == '\0') {
      break;
    }

    last_ty = ty;
    ++pos;
  }

  rv.push_back(Token{L_TOKEN_PUNCTUATION, ")"});

  return rv;
}

class Tokenizer {
  std::vector<Token> tokens_;
  size_t i_;

 public:
  // TODO: (penguinliong) Better error handling?
  Tokenizer(const std::string& lit) : tokens_(lex(lit.c_str())), i_() {}

  inline const bool empty() const { return tokens_.size() <= i_; }
  inline const Token& peak() const {
    if (empty()) {
      throw std::logic_error("tokenizer is empty");
    }
    return tokens_[i_];
  }
  inline Token next() {
    if (empty()) {
      throw std::logic_error("tokenizer is empty");
    }
    return std::move(tokens_[i_++]);
  }

  inline bool next_is_symbol() const { return peak().ty == L_TOKEN_SYMBOL; }
  inline bool next_is_integer() const { return peak().ty == L_TOKEN_INTEGER; }
  inline bool next_is_punctuation(const std::string& punc = "") const {
    return peak().ty == L_TOKEN_PUNCTUATION && (punc.empty() || peak().punctuation == punc);
  }
};

enum AstNodeType {
  L_AST_CONSTANT,
  L_AST_SYMBOL,
  L_AST_SUBNODE,
};
enum AstDataType {
  L_AST_FLOAT,
  L_AST_UINT,
};

struct Ast {
  AstNodeType node_ty;

  // L_AST_CONSTANT
  int32_t constant;
  float constant_f;
  int range_hint_high;

  // L_AST_SYMBOL
  std::string symbol;

  // L_AST_SUBNODE
  std::string op;
  std::shared_ptr<Ast> left;
  std::shared_ptr<Ast> right;
  AstDataType dtype = L_AST_UINT;

  static std::shared_ptr<Ast> make_constant(int constant) {
    auto ast = std::make_shared<Ast>();
    ast->node_ty = L_AST_CONSTANT;
    ast->constant = constant;
    ast->dtype = L_AST_UINT;
    return ast;
  }
  static std::shared_ptr<Ast> make_constant_float(float constant) {
    auto ast = std::make_shared<Ast>();
    ast->node_ty = L_AST_CONSTANT;
    ast->constant_f = constant;
    ast->dtype = L_AST_FLOAT;
    return ast;
  }
  static std::shared_ptr<Ast> make_symbol(const std::string& symbol) {
    auto ast = std::make_shared<Ast>();
    ast->node_ty = L_AST_SYMBOL;
    ast->symbol = symbol;
    ast->range_hint_high = 0;
    return ast;
  }
  static std::shared_ptr<Ast> make_node(const std::string& op, std::shared_ptr<Ast> left,
                                        std::shared_ptr<Ast> right,
                                        AstDataType dtype = L_AST_UINT) {
    auto ast = std::make_shared<Ast>();
    ast->node_ty = L_AST_SUBNODE;
    ast->op = op;
    ast->left = std::forward<std::shared_ptr<Ast>>(left);
    ast->right = std::forward<std::shared_ptr<Ast>>(right);
    ast->dtype = dtype;
    return ast;
  }

  inline bool is_constant() const { return node_ty == L_AST_CONSTANT; }
  inline bool is_node(const std::string& expected_op = "") const {
    return node_ty == L_AST_SUBNODE && (expected_op.empty() || expected_op == op);
  }
  inline bool is_associative_node() const { return is_node("*") || is_node("/") || is_node("%"); }
  inline bool is_combinational_node() const { return is_node("+"); }
  inline bool is_symbol() const { return node_ty == L_AST_SYMBOL; }
  inline bool is_constexpr() const {
    return is_constant() || (is_node() && left->is_constant() && right->is_constant());
  }
};

std::shared_ptr<Ast> parse_expr(Tokenizer& tokenizer);

std::shared_ptr<Ast> parse_factor(Tokenizer& tokenizer) {
  if (tokenizer.next_is_integer()) {
    auto token = tokenizer.next();
    return Ast::make_constant(token.integer);
  }
  if (tokenizer.next_is_symbol()) {
    auto token = tokenizer.next();
    return Ast::make_symbol(token.symbol);
  }
  if (tokenizer.next_is_punctuation("(")) {
    tokenizer.next();
    auto ast = parse_expr(tokenizer);
    if (tokenizer.next_is_punctuation(")")) {
      tokenizer.next();
    } else {
      throw std::logic_error("expected `(`, found other token");
    }
    // Sounds like a type cast.
    bool is_type_cast =
        ast->is_symbol() && (tokenizer.next_is_integer() || tokenizer.next_is_symbol() ||
                             tokenizer.next_is_punctuation("("));
    if (is_type_cast) {
      if (ast->symbol == "int") {
        ast = parse_factor(tokenizer);
      } else {
        throw std::logic_error("unsupported type cast");
      }
    }
    return ast;
  }
  throw std::logic_error("unexpected token or end of input");
}
std::shared_ptr<Ast> parse_term(Tokenizer& tokenizer) {
  std::shared_ptr<Ast> left = parse_factor(tokenizer);
  while (!tokenizer.empty()) {
    auto match = tokenizer.next_is_punctuation("*") || tokenizer.next_is_punctuation("/") ||
                 tokenizer.next_is_punctuation("<<") || tokenizer.next_is_punctuation(">>") ||
                 tokenizer.next_is_punctuation("%");
    if (!match) {
      return left;
    }

    auto op_token = tokenizer.next();
    auto right = parse_factor(tokenizer);

    left = Ast::make_node(op_token.punctuation, std::move(left), std::move(right));
  }
  return left;
}
std::shared_ptr<Ast> parse_expr(Tokenizer& tokenizer) {
  std::shared_ptr<Ast> left = parse_term(tokenizer);

  while (!tokenizer.empty()) {
    auto match = tokenizer.next_is_punctuation("+");
    if (!match) {
      return left;
    }

    auto op_token = tokenizer.next();
    auto right = parse_term(tokenizer);

    left = Ast::make_node(op_token.punctuation, std::move(left), std::move(right));
  }
  return left;
}

void print_impl(std::stringstream& ss, const std::shared_ptr<Ast>& ast) {
  if (ast->node_ty == L_AST_CONSTANT) {
    if (ast->dtype == L_AST_FLOAT) {
      ss << ast->constant_f;
    } else {
      ss << ast->constant;
    }
  } else if (ast->node_ty == L_AST_SYMBOL) {
    ss << ast->symbol;
  } else {
    bool need_paren =
        ast->op != "*" && ast->op != "/" && ast->op != "%" && ast->op != "<<" && ast->op != ">>";
    if (ast->dtype == L_AST_FLOAT || ast->left->dtype == L_AST_FLOAT ||
        ast->right->dtype == L_AST_FLOAT) {
      ss << "(int)";
    }
    ss << "(";
    if (ast->op == ">>" || ast->op == "<<") {
      ss << "(int)";
    }
    print_impl(ss, ast->left);
    ss << " " << ast->op << " ";
    print_impl(ss, ast->right);
    ss << ")";
  }
}

std::string print(const std::shared_ptr<Ast>& ast) {
  std::stringstream ss;
  print_impl(ss, ast);
  return ss.str();
}
const char* Dump(const std::shared_ptr<Ast>& ast) {
  static std::string debug;
  debug = print(ast);
  // std::cout << debug << "\n";
  return debug.c_str();
}
void hint_symbol(std::shared_ptr<Ast>& ast, const std::string& symbol, int high) {
  if (ast->is_symbol() && ast->symbol == symbol) {
    ast->range_hint_high = high;
  }
  if (ast->is_node()) {
    hint_symbol(ast->left, symbol, high);
    hint_symbol(ast->right, symbol, high);
  }
}

// Move constant coefficients to the left.
void simplify_prioritize_mul_coefficients(std::shared_ptr<Ast>& ast) {
  if (ast->is_node()) {
    simplify_prioritize_mul_coefficients(ast->left);
    simplify_prioritize_mul_coefficients(ast->right);
    if (ast->is_node("*")) {
      if (!ast->left->is_constant() && ast->right->is_constant()) {
        auto temp = std::move(ast->left);
        ast->left = std::move(ast->right);
        ast->right = std::move(temp);
      }
    }
  }
}

// Depends on `simplify_prioritize_mul_coefficients`.
bool simplify_is_multiple_of(std::shared_ptr<Ast>& ast, int divisor) {
  if (ast->is_symbol()) {
    return false;
  }
  if (ast->is_constant()) {
    return ast->constant % divisor == 0;
  }
  if (ast->is_node("*")) {
    return simplify_is_multiple_of(ast->left, divisor);
  }
  if (ast->is_combinational_node()) {
    return simplify_is_multiple_of(ast->left, divisor) &&
           simplify_is_multiple_of(ast->right, divisor);
  }
  return false;
}

// Get all coefficients in a polynomial. The function returns no element if the
// sub-expression contains any division or modulo.
bool simplify_collect_coefficients_impl(std::shared_ptr<Ast>& ast, std::vector<int>& out) {
  if (ast->is_symbol()) {
    out.push_back(1);
    return true;
  }
  if (ast->is_constant()) {
    out.push_back(ast->constant);
    return true;
  }
  if (ast->is_node("*")) {
    return simplify_collect_coefficients_impl(ast->left, out);
  }
  if (ast->is_combinational_node()) {
    return simplify_collect_coefficients_impl(ast->left, out) &&
           simplify_collect_coefficients_impl(ast->right, out);
  }
  return false;
}
std::vector<int> simplify_collect_coefficients(std::shared_ptr<Ast>& ast) {
  std::vector<int> out;
  return simplify_collect_coefficients_impl(ast, out) ? out : std::vector<int>{};
}
int simplify_get_coefficient_gcd(std::shared_ptr<Ast>& ast) {
  auto gcd = [](int a, int b) {
    if (a == 0 || b == 0) return std::max(a, b);
    while (a != b) {
      if (a > b) {
        a -= b;
      } else {
        b -= a;
      }
    }
    return a;
  };
  auto coes = simplify_collect_coefficients(ast);
  if (coes.empty()) {
    return 1;
  }

  auto out = coes[0];
  for (auto coe : coes) {
    out = gcd(coe, out);
  }
  return out;
}

// Get the upper bound of the values in this sub-expression. Retuens 0 if one
// term has never been hinted.
int simplify_upper_bound_of(const std::shared_ptr<Ast>& ast) {
  if (ast->is_constant()) {
    return ast->constant;
  }
  if (ast->is_symbol()) {
    // Can be zero, and we let it contaminate the other numbers to give a zero
    // result.
    return ast->range_hint_high;
  }
  if (ast->is_node("*")) {
    return simplify_upper_bound_of(ast->left) * simplify_upper_bound_of(ast->right);
  }
  if (ast->is_node("%")) {
    return simplify_upper_bound_of(ast->right);
  }
  if (ast->is_node("/")) {
    auto divisor = simplify_upper_bound_of(ast->right);
    return (simplify_upper_bound_of(ast->left) + divisor - 1) / divisor;
  }
  if (ast->is_node("+")) {
    return simplify_upper_bound_of(ast->left) + simplify_upper_bound_of(ast->right);
  }
  throw std::logic_error("not implemented yet");
}

// Fold multiplications.
//
// Depends on `simplify_prioritize_mul_coefficients`.
void simplify_associate_mul(std::shared_ptr<Ast>& ast) {
  if (ast->is_node()) {
    // Ensure the sub-expression has been folded.
    simplify_associate_mul(ast->left);
    simplify_associate_mul(ast->right);

    // `simplify_prioritize_mul_coefficients` ensures that all constant
    // multiplicants are on the left.
    if (ast->is_node("*") && ast->left->is_constant()) {
      // Fold constexpr.
      if (ast->right->is_constant()) {
        ast = Ast::make_constant(ast->left->constant * ast->right->constant);
        return;
      }
      // Aggregation of coefficients.
      if (ast->right->is_node("*") && ast->right->left->is_constant()) {
        ast->right->left->constant *= ast->left->constant;
        ast = std::move(ast->right);
      }
      // DO NOT support folding with division because it integral division
      // implicitly gives a floored result.
    }
  }
}

// Fold divisions. The divisors are always on the right.
//
// Depends on `simplify_prioritize_mul_coefficients`.
void simplify_associate_div_remove_nop(std::shared_ptr<Ast>& ast, int operand) {
  if (!ast) {
    return;
  }
  // if (ast->is_node()) {
  simplify_associate_div_remove_nop(ast->left, operand);
  simplify_associate_div_remove_nop(ast->right, operand);

  // Complicated cases.
  if (ast->is_node("*") && ast->left->is_constant()) {
    // if (ast->left->constant % operand == 0) {
    //  ast->left->constant /= operand;
    //}
  } else if (ast->is_constant()) {
    ast->constant /= operand;
  }
  //}
}
void simplify_associate_div(std::shared_ptr<Ast>& ast) {
  if (ast->is_node()) {
    simplify_associate_div(ast->left);
    simplify_associate_div(ast->right);

    if (ast->is_node("/") && ast->right->is_constant()) {
      // Fold constexpr.
      if (ast->left->is_constant()) {
        ast = Ast::make_constant(ast->left->constant / ast->right->constant);
        return;
      }
      // Aggregation of coefficients.
      if (ast->left->is_node("/") && ast->left->right->is_constant()) {
        ast->left->right->constant *= ast->right->constant;
        ast = std::move(ast->left);
      }
      // Fold multiply-divide patterns. Only do this when the multiplicant is a
      // multiple of the divisor. `simplify_prioritize_mul_coefficients` ensures
      // that all constant multiplicants are on the left.
      if (simplify_is_multiple_of(ast->left, ast->right->constant)) {
        simplify_associate_div_remove_nop(ast->left, ast->right->constant);
        ast = std::move(ast->left);
      }
      // THE FOLLOWING SECTION MUST PRECEDE THE ONE ABOVE; OTHERWISE THE STACK
      // WOULD OVERFLOW.
      // In case the left-expression share a common divisor, we can extract the
      // common divisor to the right-constant. In some cases, the common divisor
      // can be a coefficient in some terms in the left-expression. So some
      // multiplication can be saved.
      if (ast->op == "/") {
        auto gcd = simplify_get_coefficient_gcd(ast->left);
        if (gcd != 1 && simplify_is_multiple_of(ast->right, gcd)) {
          ast->right->constant /= gcd;
          auto left = Ast::make_node("/", std::move(ast->left), Ast::make_constant(gcd));
          simplify_associate_div(left);
          ast->left = std::move(left);
        }
      }
    }
  }
}

// Fold modulos. The divisors are always on the right.
//
// Depends on `simplify_prioritize_mul_coefficients`.
void simplify_associate_mod_remove_nop(std::shared_ptr<Ast>& ast, int operand) {
  if (ast->is_node()) {
    simplify_associate_mod_remove_nop(ast->left, operand);
    simplify_associate_mod_remove_nop(ast->right, operand);

    // Complicated cases.
    if (ast->is_node("*") && ast->left->is_constant()) {
      if (ast->left->constant % operand == 0) {
        ast = Ast::make_constant(0);
        return;
      }
    }
  }
}

void simplify_associate_mod(std::shared_ptr<Ast>& ast) {
  if (ast->is_node()) {
    simplify_associate_mod(ast->left);
    simplify_associate_mod(ast->right);

    if (ast->is_node("%") && ast->right->is_constant()) {
      // Fold constexpr.
      if (ast->left->is_constant()) {
        ast = Ast::make_constant(ast->left->constant % ast->right->constant);
        return;
      }
      // Aggregation of coefficients.
      if (ast->left->is_node("%") && ast->left->right->is_constant()) {
        ast->left->right->constant *= ast->right->constant;
        ast = std::move(ast->right);
      }
      // Fold multiply-modulo patterns. Only do this when the multiplicant is a
      // multiple of the divisor; and in that case the expression always gives a
      // zero. `simplify_prioritize_mul_coefficients` ensures that all constant
      // multiplicants are on the left.
      simplify_associate_mod_remove_nop(ast->left, ast->right->constant);
      // If the upper bound is hinted never reaching the modulo divisor, the
      // modulo can be removed.
      auto upper_bound = simplify_upper_bound_of(ast->left);
      if (upper_bound > 0 && upper_bound <= ast->right->constant) {
        ast = std::move(ast->left);
      }
    }
  }
}

// Remove nops that doesn"t have any effect on expression evaluation.
void simplify_remove_nop(std::shared_ptr<Ast>& ast) {
  if (ast->is_node()) {
    simplify_remove_nop(ast->left);
    simplify_remove_nop(ast->right);

    // Nops.
    if (ast->is_node("*")) {
      if (ast->right->is_constant() && ast->right->constant == 1) {
        ast = std::move(ast->right);
        return;
      }  // calculate all constant
      else if (ast->right->is_constant() && ast->left->is_constant()) {
        ast = Ast::make_constant(ast->left->constant * ast->right->constant);
        return;
      }
      // Reduce to zero.
      else if (ast->right->is_constant() && ast->right->constant == 0) {
        ast = Ast::make_constant(0);
        return;
      }
    }
    if (ast->is_node("/")) {
      if (ast->right->is_constant() && ast->right->constant == 1) {
        ast = std::move(ast->left);
        return;
      } else if (ast->right->is_constant() && ast->left->is_constant()) {
        ast = Ast::make_constant(ast->left->constant / ast->right->constant);
        return;
      }
    }
    if (ast->is_node("+")) {
      if (ast->right->is_constant() && ast->right->constant == 0) {
        ast = std::move(ast->left);
        return;
      }
      if (ast->left->is_constant() && ast->left->constant == 0) {
        ast = std::move(ast->right);
        return;
      }
      if (ast->right->is_constant() && ast->left->is_constant()) {
        ast = Ast::make_constant(ast->left->constant + ast->right->constant);
        return;
      }
    }
    if (ast->is_node("%")) {
      if (ast->right->is_constant() && ast->right->constant == 0) {
        throw std::runtime_error("remainder can't be zero ");
      }
      if (ast->right->is_constant() && ast->right->constant == 1) {
        ast = std::move(ast->left);
        return;
      }
      if (ast->left->is_constant() && ast->left->constant == 0) {
        ast = Ast::make_constant(0);
        return;
      }
      if (ast->right->is_constant() && ast->left->is_constant()) {
        ast = Ast::make_constant(ast->left->constant % ast->right->constant);
        return;
      }
      // Reduce to zero.
      if (ast->right->is_constant() && ast->right->constant == 1) {
        ast = Ast::make_constant(0);
        return;
      }
    }
  }
}

// replace div and mod with bit_shift and mul
void simplify_mod_and_div_with_bitop(std::shared_ptr<Ast>& ast) {
  if (ast->is_node()) {
    simplify_mod_and_div_with_bitop(ast->left);
    simplify_mod_and_div_with_bitop(ast->right);
    if (ast->is_node("/")) {
      if (ast->right->is_constant()) {
        if (is_expon_by_2(ast->right->constant)) {
          ast = Ast::make_node(">>", std::move(ast->left),
                               Ast::make_constant(mylog(ast->right->constant)));
          return;
        } else {
          // option 1: a/const_b ==? a*1/const_b used now
          // option 2: a/const_b ==? int(a*128 *(1/const_b))>>7
          // option 3:
          //    int get_min_bits(int b, uint32_t & c) {
          //        int i = 0;
          //        for (i = 0; true; i++) {
          //            c = b - ((uint64_t)1 << (32 + i)) % b;
          //            if (c == b) {
          //                c = 0;//b%2==0
          //                break;
          //            }
          //            else if (is2exp(c)) {
          //                if (log2x(c) <= i) break;
          //            }
          //            else {
          //                if (log2x(c) < i) break;
          //            }
          //        }
          //        return i;
          //    }

          //    uint32_t b,c=0,i;
          //    b = 3154;//divisor,b > 1
          //    uint32_t e, db;
          //    i = get_min_bits(b, c);
          //    e = log2x(b);
          //    db = (((uint64_t)1 << (32 + i)) + c) / b;//when i = e+1 get low 32bit
          //    if (i <= e) {
          //        printf("quotient  = ((uint64_t)dividend * %d) >> %d\n", db, 32 + i);
          //    }
          //    else {//i = e + 1 means e <= floor(log2(b)) can't satify c*a < 2^(e+32)
          //        printf("t = ((uint64_t)dividend * %d) >> 32\n", db);
          //        printf("quotient  = ((((uint64_t)dividend - t) >> 1) + t) >> %d\n", e);
          //    }

          // int type
          // uint32_t bediv = get_nearsest_expo_by2(ast->right->constant);
          // float new_bediv = 1.0 / ast->right->constant * bediv;
          // float type
          // ast->left = Ast::make_node('*', std::move(ast->left), Ast::make_constant(new_bediv),
          // L_AST_FLOAT); ast = Ast::make_node(get_one_operation(">>"), std::move(ast->left),
          // Ast::make_constant(mylog(bediv)));
          uint32_t bediv = 1;
          // float type
          float new_bediv = 1.0 / ast->right->constant * bediv;
          ast = Ast::make_node("*", std::move(ast->left), Ast::make_constant_float(new_bediv));
          return;
        }
      }
    }
    if (ast->is_node("%")) {
      if (ast->right->is_constant() && is_expon_by_2(ast->right->constant)) {
        ast = Ast::make_node("&", std::move(ast->left),
                             Ast::make_constant(ast->right->constant-1));
        return;
      } else {
        // float type
        // shared_ptr can't have another copy, so we can't simplify a%b to 
        // a%b => a-int(a/b)*b => a-int(a*(1/b))*b, notice b is constant
        float new_bediv = (1.0 / ast->right->constant);
        auto r = ast->right;
        ast->right = Ast::make_node("*", (ast->left), Ast::make_constant_float(new_bediv));
        ast->right = Ast::make_node("*", (ast->right), r);
        ast = Ast::make_node("-", (ast->left), ast->right);
        return;
      }
    }
  }
}

void simplify(std::shared_ptr<Ast>& ast) {
  // DON'T CHANGE THE ORDER HERE.
  simplify_prioritize_mul_coefficients(ast);
  simplify_associate_mul(ast);
  simplify_associate_div(ast);
  simplify_associate_mod(ast);
  simplify_remove_nop(ast);
  simplify_mod_and_div_with_bitop(ast);
}

#define ICHECKEQ(actual, expected, expr_lit)                                         \
  {                                                                                  \
    if (actual == expected) {                                                        \
      std::cout << "[PASS] " << expr_lit << " ==> " << expected << "\n";             \
    } else {                                                                         \
      std::cout << __LINE__ << ":[FAIL] " << expr_lit << " failed.\n got " << actual \
                << " vs wanted " << expected << "\n";                                \
      assert(0);                                                                    \
    }                                                                                \
  }
void test_folder_div() {
  {
    const char* expr_lit = "0/2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "0", expr_lit);
  }
  {
    const char* expr_lit = "1/2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "0", expr_lit);
  }
  {
    const char* expr_lit = "2/2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "1", expr_lit);
  }
  {
    const char* expr_lit = "4/2/2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "1", expr_lit);
  }
  {
    const char* expr_lit = "5/2/2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "1", expr_lit);
  }
  {
    const char* expr_lit = "3/2/2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "0", expr_lit);
  }
  {
    const char* expr_lit = "(((((group_id0)*16)+((local_id0)*8+4))))/8";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "((int)((4 * group_id0) + ((2 * local_id0) + 1)) >> 1)", expr_lit);
  }
  {
    const char* expr_lit = "(((((group_id0)*16)+((local_id0)*8+4))))/4";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "((4 * group_id0) + ((2 * local_id0) + 1))", expr_lit);
  }
  {
    const char* expr_lit = "(((((group_id0)*16)+((local_id0)*8+4+0))))/4";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "((4 * group_id0) + ((2 * local_id0) + 1))", expr_lit);
  }
  {
    const char* expr_lit = "(((((group_id0)*16)+((local_id0)*8+5))))/4";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "((int)((16 * group_id0) + ((8 * local_id0) + 5)) >> 2)", expr_lit);
  }
  {
    const char* expr_lit = "(((((group_id0)*16)+((local_id0)*9+4))))/4";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "((int)((16 * group_id0) + ((9 * local_id0) + 4)) >> 2)", expr_lit);
  }
  {
    const char* expr_lit = "(((((group_id0)*16)+((local_id0)*9+4))))/5";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "(int)(((16 * group_id0) + ((9 * local_id0) + 4)) * 0.2)", expr_lit);
  }
}

void test_folder_mod() {
  {
    const char* expr_lit = "0%2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "0", expr_lit);
  }
  {
    const char* expr_lit = "1%2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "1", expr_lit);
  }
  {
    const char* expr_lit = "2%2";
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    ICHECKEQ(print(ast), "0", expr_lit);
  }
}

void test_all() {
  test_folder_div();
  test_folder_mod();
}


std::string DoSimplify(const std::string& expr_lit_c) {
  try {
    std::string expr_lit = std::regex_replace(expr_lit_c, std::regex("\\(int\\)"), "_int_");
    expr_lit = std::regex_replace(expr_lit, std::regex("\\."), "_dot_");
    Tokenizer tokenizer(expr_lit);
    auto ast = parse_expr(tokenizer);
    simplify(ast);
    expr_lit = print(ast);
    expr_lit = std::regex_replace(expr_lit, std::regex("_int_"), "(int)");
    expr_lit = std::regex_replace(expr_lit, std::regex("_dot_"), ".");
    return expr_lit;
  } catch (std::exception& e) {
    LOG(WARNING) << " expression simplify Exception:" << e.what()
                 << ".current expr=" << expr_lit_c;

    return expr_lit_c;
  }
}

int __main(int argc, const char** argv) {
  const char* expr_lit =
      "(((((((group_id2)*1048576)+((local_id2)*16384))+((group_id1)*1024))+((local_id1)*512))+(("
      "group_id0)*32))+((local_id0)*8))/4/64";
  // const char* expr_lit = "143/132/2";

  Tokenizer tokenizer(expr_lit);
  auto ast = parse_expr(tokenizer);
  hint_symbol(ast, "global_id0", 2);
  hint_symbol(ast, "local_id0", 2);
  simplify(ast);
  std::cout << "input: " << expr_lit << std::endl;
  std::cout << "ouptut: " << print(ast) << std::endl;
  std::cout << std::endl;
  test_all();
  return 0;
}

}  // namespace exprSimp
}  // namespace tir
}  // namespace tvm