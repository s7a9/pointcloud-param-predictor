#pragma once

#include <iostream>

#include "algorithm/autograd/autograd.h"

using namespace autograd;

inline int autograd_test() {
	Variable v1(3, 3);
	v1 << 1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	Variable v2(3, 3);
	v2 << 1, 2, 3,
		4, 5, 6,
		7, 8, 9;
	matrix_t mat(2, 2); mat << 1, 2, 3, 4;
	auto ii = mat.row(0);
	function::WeightedAdd add({ 0.1, 0.2 });
	function::Multiply mul;
	auto v3 = add.forward({ &v1, &v2 });
	auto v4 = mul.forward({ &v1, &v3 });
	v4.backward(matrix_t::Ones(3, 3));
	std::cout << v4 << std::endl;
	std::cout << v3.grad() << std::endl;
	std::cout << v1.grad() << std::endl;
	std::cout << v2.grad() << std::endl;
	return 0;
}