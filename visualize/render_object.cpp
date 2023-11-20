#include "render_object.h"

#include <Windows.h>
#include <GL/gl.h>
#include <FL/gl.h>

#include <algorithm/pointcloud/fileio.h>

namespace visual {

void Axes::render() {
	glBegin(GL_LINES);
	gl_color(pos_clr_);
	glVertex3f(0, 0, 0); glVertex3f(length_, 0, 0);
	glVertex3f(0, 0, 0); glVertex3f(0, length_, 0);
	glVertex3f(0, 0, 0); glVertex3f(0, 0, length_);
	gl_color(neg_clr_);
	glVertex3f(0, 0, 0); glVertex3f(-length_, 0, 0);
	glVertex3f(0, 0, 0); glVertex3f(0, -length_, 0);
	glVertex3f(0, 0, 0); glVertex3f(0, 0, -length_);
	glEnd();
}

void PointCloud::render() {
	if (pointcloud_ == nullptr) return;
	auto& pointcloud = *pointcloud_;
	glBegin(GL_POINTS);
	gl_color(clr_);
	for (int i = 0; i < pointcloud.rows(); ++i) {
		glVertex3f(pointcloud(i, 0), pointcloud(i, 1), pointcloud(i, 2));
	}
	glEnd();
}

}