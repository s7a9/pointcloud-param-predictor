#pragma once

#include <vector>
#include <optional>
#include <Eigen/Dense>

#include <algorithm/autograd/config.h>

namespace autograd {

class Variable;

class Function {
public:
    inline void inference() { with_grad_ = false; }

    inline void train() { with_grad_ = true; }

    virtual Variable forward(const std::vector<Variable*>& input) const = 0;

    virtual void backward(const Variable* father, const std::vector<Variable*>& children) const = 0;

    template <class FuncType>
    FuncType* as() { return dynamic_cast<FuncType*>(this); }

protected:
    bool with_grad_ = true;

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
    MatMul() = default;

    explicit MatMul(matrix_t&& mat): mat_(mat) {}

    inline void set_mat(matrix_t&& mat) { mat_ = mat; }

    Variable forward(const std::vector<Variable*>& input) const;

    void backward(const Variable* father, const std::vector<Variable*>& children) const;

private:
    std::optional<matrix_t> mat_;
};

class Multiply : public Function {
public:
    Variable forward(const std::vector<Variable*>& input) const;

    void backward(const Variable* father, const std::vector<Variable*>& children) const;
};

class MakeScaleMat : public Function {
public:
	MakeScaleMat() = default;

	explicit MakeScaleMat(size_t mat_size): mat_size_(mat_size) {}

	inline void set_mat_size(size_t mat_size) { mat_size_ = mat_size; }

	Variable forward(const std::vector<Variable*>& input) const;

	void backward(const Variable* father, const std::vector<Variable*>& children) const;

private:
    size_t mat_size_ = 0;
};

}