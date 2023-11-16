#pragma once

#include <vector>
#include <Eigen/Dense>

#include <algorithm/autograd/config.h>

namespace autograd {

class Variable;

class Function {
public:
    virtual Variable forward(const std::vector<Variable*>& input) const = 0;

    virtual void backward(const Variable* father, const std::vector<Variable*>& children) const = 0;

protected:
    static void add_deg_out(const std::vector<Variable*>& children);
};

}

namespace autograd::function {

class Add : public Function {
public:
    Variable forward(const std::vector<Variable*>& input) const;

    void backward(const Variable* father, const std::vector<Variable*>& children) const;
};

class WeightedAdd : public Function {
public:
    WeightedAdd(std::vector<dtype>&& weights):
        weights_(weights) {}

    Variable forward(const std::vector<Variable*>& input) const;

    void backward(const Variable* father, const std::vector<Variable*>& children) const;

private:
    std::vector<dtype> weights_;
};

class MatMul : public Function {
public:
    MatMul(matrix_t&& mat):
        mat_(mat) {}

    Variable forward(const std::vector<Variable*>& input) const;

    void backward(const Variable* father, const std::vector<Variable*>& children) const;

private:
    matrix_t mat_;
};

class Multiply : public Function {
public:
    Variable forward(const std::vector<Variable*>& input) const;

    void backward(const Variable* father, const std::vector<Variable*>& children) const;
};

}