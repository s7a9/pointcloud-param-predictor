#pragma once

#include <algorithm/autograd/autograd.h>

namespace pointcloud {

using pointcloud_t = autograd::matrix_t;

bool is_pointcloud(const autograd::matrix_t& matrix);

/// The Earth Mover's Distance between two point clouds.
pointcloud_t EMDLoss(
	const pointcloud_t& predicted,
	const pointcloud_t& target
);

/// The Chamfer Distance between two point clouds.
pointcloud_t ChamferLoss(
	const autograd::matrix_t& predicted,
	const pointcloud_t& target
);

}

#include "generator.h"