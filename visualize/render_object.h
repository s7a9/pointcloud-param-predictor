#pragma once

#include <FL/Fl.H>

#include <algorithm/pointcloud/pointcloud.h>

namespace visual {

class RenderObject {
public:
	RenderObject() = default;
	virtual ~RenderObject() = default;

	virtual void render() = 0;
private:
};

class Axes : public RenderObject {
public:
	Axes(float length = 10000.f, Fl_Color pos_clr = FL_RED, Fl_Color neg_clr = FL_BLUE)
		: length_(length), pos_clr_(pos_clr), neg_clr_(neg_clr) {}

	~Axes() = default;

	void render();

private:
	float length_;

	Fl_Color pos_clr_, neg_clr_;
};

class PointCloud : public RenderObject {
public:
	PointCloud(pointcloud::pointcloud_t* pointcloud, Fl_Color clr) :
		pointcloud_(pointcloud), clr_(clr) {}

	void render();

private:
	pointcloud::pointcloud_t* pointcloud_;

	Fl_Color clr_;
};

}