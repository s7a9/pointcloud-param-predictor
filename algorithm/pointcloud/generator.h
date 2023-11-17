#pragma once

#include <utility>
#include <functional>

#include <algorithm/autograd/variable.h>

#include "pointcloud.h"

namespace pointcloud {

class ModelBase {
public:
	/// <summary>
	/// generate a pointcloud
	/// </summary>
	/// <param name="sample_num">[IN] how many point will be generated</param>
	/// <returns>a pointcloud</returns>
	virtual pointcloud_t& generate(size_t sample_num) = 0;

	virtual ~ModelBase() {
		for (auto function : functions_)
			delete(function);
	}

	inline void inference() {
		for (auto function : functions_)
			function->inference();
	}

	inline void train() {
		for (auto function : functions_)
			function->train();
	}

	inline void zero_grads() {
		variables_.clear();
	}

protected:
	inline void add_func(autograd::Function* function) {
		functions_.push_back(function);
	}

	inline void add_var(autograd::Variable&& variable) {
		variables_.push_back(variable);
	}

	std::vector<autograd::Function*> functions_;

	std::vector<autograd::Variable> variables_;
};

class LineModel : public ModelBase {
public:
	/// end_points is a 2x3 matrix, each row is a point
	LineModel(autograd::matrix_t&& end_points);

	pointcloud_t& generate(size_t sample_num);

private:
	autograd::Variable end_points_;
};;

}