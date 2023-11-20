#pragma once

#include <map>

#include <Windows.h>
#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Text_Editor.H>
#include <FL/gl.h>

#include <algorithm/pointcloud/fileio.h>
#include <algorithm/pointcloud/generator.h>
#include "render_object.h"

namespace visual {


class GlWindow : public Fl_Gl_Window {
public:
	GlWindow(int x, int y, int w, int h, const char* l = 0);

	~GlWindow();

	void set_center_point(float x, float y, float z);

	void set_lookat_angle_delta(float dtheta, float dphi);

	void reset_lookat_angle() { lookat_theta_ = lookat_phi_ = .0f; }

	void set_lookat_dis_delta(float ddis);

	int handle(int event);

	void add_object(const char* name, RenderObject* object);

	void remove_object(const char* name);

	void optimize_step(float lr);

protected:
	void fix_viewport(int w, int h);

	void resize(int, int, int, int);

	void draw();

	void set_lookat();

	float cp_x_, cp_y_, cp_z_;

	float lookat_theta_, lookat_phi_, lookat_distance_;

	std::map<std::string, RenderObject*> render_objects_;

	pointcloud::pointcloud_t target_;

	pointcloud::PlaneModel plane_;
};

class AppWindow : public Fl_Window {
public:
	AppWindow(int w, int h, const char* l = 0);

private:

	static void set_center_point_btn_pushed_callback_(Fl_Widget* widget, void* data);

	static void fitting_step_btn_pushed_callback_(Fl_Widget* widget, void* data);

	GlWindow* gl_window_;

	Fl_Text_Buffer* center_point_text_buffer_[3];

	Fl_Text_Editor* center_point_editor_[3];

	Fl_Button* set_center_point_btn_;

	Fl_Button* fitting_step_btn_;
};

}