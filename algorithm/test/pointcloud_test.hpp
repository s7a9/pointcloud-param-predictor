#pragma once

#include <iostream>

#include <algorithm/pointcloud/fileio.h>
#include <algorithm/pointcloud/generator.h>

inline int pointcloud_test() {
	using namespace pointcloud;
	using namespace autograd;
	matrix_t end_points(2, 3);
	end_points << 0, 0, 0,
		1, 1, 1;
	LineModel line(std::move(end_points));
	auto& pc = line.generate(20);
	std::cout << pc << std::endl;
	save_pointcloud("pc_test.bin", pc);
	auto pc2 = load_pointcloud("pc_test.bin");
	std::cout << "========\n" << pc2 << std::endl;
	return 0;
}