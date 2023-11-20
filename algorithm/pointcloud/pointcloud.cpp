#include "pointcloud.h"

#include <iostream>

namespace pointcloud {

static autograd::matrix_t
pairwise_distance(const autograd::matrix_t& x, const autograd::matrix_t& y) {
	autograd::matrix_t dist(x.rows(), y.rows());
	for (int i = 0; i < x.rows(); ++i) {
		for (int j = 0; j < y.rows(); ++j) {
			dist(i, j) = (x.row(i) - y.row(j)).norm();
		}
	}
	return dist;
}

/* Compute the Earth Mover's Distance (or Wasserstein metric).
 * EMD is the minimum cost of moving the mass of one point cloud to the other.
 * It can be seems as as linear optimization problem. For the math part, please
 * refer to `readme.md`.
 * Here we use the **Simplex Algorithm** to solve the linear optimization problem.
 */

pointcloud_t ChamferLoss(const autograd::matrix_t& predicted, const pointcloud_t& target) {
	pointcloud_t loss1 = autograd::matrix_t::Zero(predicted.rows(), 3);
	pointcloud_t loss2 = autograd::matrix_t::Zero(predicted.rows(), 3);
	autograd::matrix_t dist = pairwise_distance(predicted, target);
	int n = predicted.rows(), m = target.rows();
	std::vector<int> row_min_index(n);
	std::vector<int> col_min_index(m);
	for (int i = 0; i < n; ++i) {
		row_min_index[i] = std::min_element(dist.row(i).begin(), dist.row(i).end()) - dist.row(i).begin();
	}
	for (int j = 0; j < m; ++j) {
		col_min_index[j] = std::min_element(dist.col(j).begin(), dist.col(j).end()) - dist.col(j).begin();
	}
	for (int i = 0; i < n; ++i) {
		loss1.row(i) = predicted.row(i) - target.row(row_min_index[i]);
	}
	for (int i = 0; i < m; ++i) {
		loss2.row(col_min_index[i]) = predicted.row(col_min_index[i]) - target.row(i);
	}
	// std::cout << loss1 << "\n!!!!!!\n" << loss2 << std::endl;
	return loss1 / (2 * n) + loss2 / (2 * m);
}

}