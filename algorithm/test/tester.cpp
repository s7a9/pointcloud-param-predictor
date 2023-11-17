#include "pointcloud_test.hpp"
#include "autograd_test.hpp"

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