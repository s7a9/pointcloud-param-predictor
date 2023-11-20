#pragma once

#include <map>
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
	virtual autograd::Variable& generate(size_t sample_num) = 0;

	virtual ~ModelBase() {
		for (auto function : functions_)
			delete(function);
		for (auto& iter : parameters_) {
			delete iter.second;
		}
	}

	inline void inference() {
		for (auto function : functions_)
			function->inference();
	}

	inline void train() {
		for (auto function : functions_)
			function->train();
	}

	inline void optimize(float lr) {
		for (auto& iter : parameters_) {
			*iter.second -= lr * iter.second->grad();
		}
	}

	inline void zero_grads() {
		variables_.clear();
		for (auto& iter : parameters_) {
			iter.second->grad().setZero();
		}
	}

	inline autograd::Variable* param(const std::string& name) {
		auto result = parameters_.find(name);
		if (result == parameters_.end())
			return nullptr;
		return result->second;
	}

protected:
	inline void add_func(autograd::Function* function) {
		functions_.push_back(function);
	}

	inline void add_var(autograd::Variable&& variable) {
		variables_.push_back(variable);
	}

	inline void add_param(autograd::Variable* param, const std::string& name) {
		parameters_[name] = param;
	}

	template <class func_t>
	inline func_t* func(int index) { return functions_[index]->as<func_t>(); }

	inline auto var(int index) { return &variables_[index]; }

	std::vector<autograd::Function*> functions_;

	std::vector<autograd::Variable> variables_;

	std::map<std::string, autograd::Variable*> parameters_;
};

class LineModel : public ModelBase {
public:
	/// end_points is a 2x3 matrix, each row is a point
	LineModel(autograd::matrix_t&& end_points);

	autograd::Variable& generate(size_t sample_num);
};

class PlaneModel : public ModelBase {
public:
	PlaneModel(autograd::matrix_t&& span, autograd::matrix_t&& cornor);

	autograd::Variable& generate(size_t sample_num);
};

}