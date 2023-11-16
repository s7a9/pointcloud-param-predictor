#include "function.h"
#include "variable.h"

namespace autograd {

void Function::add_deg_out(const std::vector<Variable*>& children) {
    for (auto var : children) { ++var->deg_out_; }
}

}

namespace autograd::function {

Variable Add::forward(const std::vector<Variable*>& input) const {
    add_deg_out(input);
    matrix_t result = matrix_t::Zero(input[0]->data().rows(), input[0]->data().cols());
    for (auto var: input) {
        result += var->data();
    }
    return Variable(std::move(result), this, true, input);
}

void Add::backward(const Variable* father, const std::vector<Variable*>& children) const {
    for (auto child: children) {
        child->backward(father->grad());
    }
}

Variable WeightedAdd::forward(const std::vector<Variable*>& input) const {
    if (input.size() != weights_.size()) {
        throw std::logic_error("WeightedAdd::forward - input size must equal weight size");
    }
    add_deg_out(input);
    matrix_t result{weights_[0] * input[0]->data()};
    for (int i = 1; i < input.size(); ++i) {
        result += weights_[i] * input[i]->data();
    }
    return Variable(std::move(result), this, true, input);
}

void WeightedAdd::backward(const Variable* father, const std::vector<Variable*>& children) const {
    for (int i = 0; i < weights_.size(); ++i) {
        children[i]->backward(father->grad() / weights_[i]);
    }
}

Variable MatMul::forward(const std::vector<Variable*>& input) const {
    if (input.size() != 1) {
        throw std::logic_error("MatMul::forward - only accept one input");
    }
    add_deg_out(input);
    return Variable(mat_ * input[0]->data(), this, true, input);
}

void MatMul::backward(const Variable* father, const std::vector<Variable*>& children) const {
    children[0]->backward(mat_.transpose() * father->grad());
}

Variable Multiply::forward(const std::vector<Variable*>& input) const {
    if (input.size() != 2) {
        throw std::logic_error("Multiply::forward - only accept two input");
    }
    add_deg_out(input);
    return Variable(input[0]->data() * input[1]->data(), this, true, input);
}

void Multiply::backward(const Variable* father, const std::vector<Variable*>& children) const {
    children[0]->backward(father->grad() * children[1]->data().transpose());
    children[1]->backward(children[0]->data().transpose() * father->grad());
}

}