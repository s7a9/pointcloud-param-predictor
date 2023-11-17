#pragma once

#include <exception>

namespace autograd {

using dtype = float;

using matrix_t = Eigen::Matrix<dtype, Eigen::Dynamic, Eigen::Dynamic>;

}