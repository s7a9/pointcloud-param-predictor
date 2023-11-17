#pragma once

#include "pointcloud.h"

namespace pointcloud {

void save_pointcloud(const char* filename, const pointcloud_t& pointcloud);

pointcloud_t load_pointcloud(const char* filename);

}