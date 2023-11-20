#pragma once

#include <optional>

#include <Eigen/Dense>
#include <algorithm/autograd/config.h>
#include <algorithm/autograd/function.h>

namespace autograd {

class Variable : public matrix_t {
    friend Function;

public:
    Variable(
        int rown, int coln,
        bool requires_grad = true
    ) : matrix_t(rown, coln),
        deg_out_(0),
        requires_grad_(requires_grad),
        grad_fn_(nullptr) {
        if (requires_grad) {
            grad_ = matrix_t::Zero(rows(), cols());
        }
    }

    Variable(
        matrix_t&& data,
        bool requires_grad = true
    ):  matrix_t(data),
        deg_out_(0),
        requires_grad_(requires_grad),
        grad_fn_(nullptr) {
        if (requires_grad) {
            grad_ = matrix_t::Zero(rows(), cols());
        }
    }

    Variable(
        matrix_t&& data,
        const Function* grad_fn,
        bool requires_grad,
        const std::vector<Variable*>& children
    ):  matrix_t(std::move(data)),
        requires_grad_(requires_grad),
        grad_fn_(grad_fn),
        deg_out_(0),
        children_(children) {
        if (requires_grad_) {
            grad_ = matrix_t::Zero(rows(), cols());
        }
    }

    Variable(Variable&& other) = default;

    Variable(const Variable& other) = default;

    ~Variable() = default;

    /// @brief Compute the gradient of this variable. Clear all gradient after backward.
    inline void backward(const matrix_t& grad) {
        if (!requires_grad_) return;
        *grad_ += grad;
        if (grad_fn_ == nullptr) return;
        if (deg_out_ > 0) --deg_out_;
        if (deg_out_ == 0) {
            grad_fn_->backward(this, children_);
            grad_->setZero();
        }
    }

    inline bool requires_grad() const { return requires_grad_; }

    inline const matrix_t& data() const { return *this; }

    inline const bool has_grad() const { return grad_.has_value(); }

    inline matrix_t& grad() { return grad_.value(); }

    inline const matrix_t& grad() const { return grad_.value(); }

    inline const std::vector<Variable*>& children() const { return children_; }

private:
    int deg_out_;
    std::optional<matrix_t> grad_;
    bool requires_grad_;
    std::vector<Variable*> children_;
    const Function* grad_fn_;
};

}