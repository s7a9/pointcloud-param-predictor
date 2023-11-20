#include "generator.h"

#include <algorithm/autograd/function.h>

namespace pointcloud {

using namespace autograd;

LineModel::LineModel(matrix_t&& end_points) {
	add_func(new function::MatMul());
	add_param(new Variable(std::move(end_points)), "end_points");
}

autograd::Variable& LineModel::generate(size_t sample_num) {
	matrix_t random_params(sample_num, 2);
	for (int i = 0; i < sample_num; ++i) {
		random_params(i, 0) = rand() / (float)RAND_MAX;
		random_params(i, 1) = 1 - random_params(i, 0);
	}
	auto mul = func<function::MatMul>(0);
	mul->set_mat(std::move(random_params));
	add_var(mul->forward({ param("end_points") }));
	return variables_.back();
}

PlaneModel::PlaneModel(matrix_t&& span, matrix_t&& cornor) {
	add_func(new function::MatMul());
	add_func(new function::MatMul());
	add_func(new function::Add());
	add_param(new Variable(std::move(span)), "span");
	add_param(new Variable(std::move(cornor)), "cornor");
}

autograd::Variable& PlaneModel::generate(size_t sample_num) {
	matrix_t random_params(sample_num, 2);
	for (int i = 0; i < sample_num; ++i) {
		random_params(i, 0) = rand() / (float)RAND_MAX;
		random_params(i, 1) = rand() / (float)RAND_MAX;
	}
	auto mul1 = func<function::MatMul>(0);
	auto mul2 = func<function::MatMul>(1);
	auto add = func<function::Add>(2);
	mul1->set_mat(matrix_t::Ones(sample_num, 1));
	mul2->set_mat(std::move(random_params));
	variables_.reserve(3);
	add_var(mul1->forward({ param("cornor") }));
	add_var(mul2->forward({ param("span") }));
	add_var(add->forward({ var(0), var(1) }));
	return variables_.back();
}

}