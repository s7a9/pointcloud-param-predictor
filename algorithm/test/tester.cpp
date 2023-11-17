#include <iostream>

#include <algorithm/autograd/autograd.h>

inline int autograd_test() {
	using namespace autograd;
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

#include <algorithm/pointcloud/fileio.h>
#include <algorithm/pointcloud/generator.h>

inline int pointcloud_test() {
	using namespace pointcloud;
	using namespace autograd;
	matrix_t end_points(2, 3);
	end_points << -5, 0, 0, 0, 5, 0;
	LineModel line(std::move(end_points));
	auto& pc = line.generate(400);
	std::cout << pc << std::endl;
	save_pointcloud("pc_test.bin", pc);
	auto pc2 = load_pointcloud("pc_test.bin");
	std::cout << "========\n" << pc2 << std::endl;
	return 0;
}

typedef int(*test_func_t)();

int main() {
	std::vector<std::pair<std::string, test_func_t>> tests = {
		{"autograd", autograd_test},
		{"pointcloud", pointcloud_test},
	};
	for (auto& test : tests) {
		std::cout << "Testing " << test.first << "..." << std::endl;
		int result;
		try {
			result = test.second();
			if (result == 0) {
				std::cout << "Test " << test.first << " passed." << std::endl;
			}
			else {
				std::cout << "Test " << test.first << " failed with code " << result << "." << std::endl;
			}
		} catch (const std::exception& e) {
			std::cerr << "Test " << test.first << " failed with exception " << e.what() << std::endl;
			result = -1;
		}
	}
	return 0;
}