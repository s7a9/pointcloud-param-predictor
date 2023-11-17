#include "fileio.h"

#include <fstream>

namespace pointcloud {

void save_pointcloud(const char* filename, const pointcloud_t& pointcloud) {
	std::ofstream out(filename, std::ios::binary);
	auto rown = pointcloud.rows();
	out.write(
		reinterpret_cast<const char*>(&rown),
		sizeof(rown)
	);
	for (int dim = 0; dim < 3; ++dim) {
		for (int i = 0; i < rown; ++i) {
			out.write(
				reinterpret_cast<const char*>(&pointcloud(i, dim)),
				sizeof(pointcloud(i, dim))
			);
		}
	}
	out.close();
}

pointcloud_t load_pointcloud(const char* filename) {
	std::ifstream in(filename, std::ios::binary);
	Eigen::Index rown = 0;
	in.read(
		reinterpret_cast<char*>(&rown),
		sizeof(rown)
	);
	pointcloud_t pointcloud(rown, 3);
	for (int dim = 0; dim < 3; ++dim) {
		for (int i = 0; i < rown; ++i) {
			in.read(
				reinterpret_cast<char*>(&pointcloud(i, dim)),
				sizeof(pointcloud(i, dim))
			);
		}
	}
	in.close();
	return pointcloud;
}

}