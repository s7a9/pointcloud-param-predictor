#include "generator.h"

#include <algorithm/autograd/function.h>

namespace pointcloud {

using namespace autograd;

LineModel::LineModel(matrix_t&& end_points) {
	add_func(new function::MatMul());
	add_param(new Variable(std::move(end_points)), "end_points");
}

autograd::Variable& LineModel::generate(size_t sample_num) {
	matrix_t random_params = matrix_t::Zero(sample_num, 2);
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
	matrix_t random_params = matrix_t::Zero(sample_num, 2);
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

CylinderModel::CylinderModel(matrix_t&& axis_root, matrix_t&& axis_dir, float radius) {
	add_func(new function::MatMul());
	add_func(new function::MatMul());
	add_func(new function::MatMul());
	add_func(new function::Add());
	add_func(new function::Add());
	add_func(new function::MakeScaleMat(3));
	add_func(new function::MatMul());
	add_param(new Variable(std::move(axis_root)), "axis_root");
	add_param(new Variable(std::move(axis_dir)), "axis_dir");
	add_param(new Variable(matrix_t::Zero(1, 1)), "radius");
	auto& param = *parameters_["radius"];
	param(0, 0) = radius;
}

autograd::Variable& CylinderModel::generate(size_t sample_num) {
	matrix_t random_axis_scaler = matrix_t::Zero(sample_num, 1);
	for (int i = 0; i < sample_num; ++i) {
		random_axis_scaler(i, 0) = rand() / (float)RAND_MAX;
	}
	matrix_t random_params = matrix_t::Random(sample_num, 3);
	auto axis_dir = param("axis_dir");
	Eigen::Vector3f axis_dir_vec = axis_dir->row(0);
	for (int i = 0; i < sample_num; ++i) {
		Eigen::Vector3f vec = random_params.row(i);
		vec = vec.cross(axis_dir_vec);
		vec.normalize();
		random_params.row(i) = vec;
	}
	auto mul1 = func<function::MatMul>(0);
	auto mul2 = func<function::MatMul>(1);
	auto mul3 = func<function::MatMul>(2);
	auto mul4 = func<function::MatMul>(6);
	auto add1 = func<function::Add>(3);
	auto add2 = func<function::Add>(4);
	auto makescale = func<function::MakeScaleMat>(5);
	mul1->set_mat(matrix_t::Ones(sample_num, 1));
	mul2->set_mat(std::move(random_axis_scaler));
	mul3->set_mat(matrix_t::Ones(3, 1));
	mul4->set_mat(std::move(random_params));
	variables_.reserve(10);
	auto var1 = add_var(mul1->forward({ param("axis_root") }));
	auto var2 = add_var(mul2->forward({ param("axis_dir") }));
	auto var3 = add_var(add1->forward({ var1, var2 }));
	auto var4 = add_var(mul3->forward({ param("radius") }));
	auto var5 = add_var(makescale->forward({ var4 }));
	auto var6 = add_var(mul4->forward({ var5 }));
	return *add_var(add2->forward({ var3, var6 }));
}

}