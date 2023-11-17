#include "generator.h"

#include <algorithm/autograd/function.h>

namespace pointcloud {

using namespace autograd;

LineModel::LineModel(matrix_t&& end_points) :
	end_points_(std::move(end_points)) {
	add_func(new function::MatMul());
}

pointcloud_t& LineModel::generate(size_t sample_num) {
	matrix_t random_params(sample_num, 2);
	for (int i = 0; i < sample_num; ++i) {
		random_params(i, 0) = rand() / (double)RAND_MAX;
		random_params(i, 1) = 1 - random_params(i, 0);
	}
	auto mul = functions_[0]->as<function::MatMul>();
	mul->set_mat(std::move(random_params));
	add_var(mul->forward({ &end_points_ }));
	return variables_.back();
}

}