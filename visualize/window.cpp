#include "window.h"

#include <GL/gl.h>
#include <GL/glu.h>

#include <string>
#include <iostream>

namespace visual {

constexpr float min_lookat_distance = 1.0f;

GlWindow::GlWindow(int x, int y, int w, int h, const char* l) :
	Fl_Gl_Window(x, y, w, h, l),
	target_(pointcloud::load_pointcloud(R"(pc_test2.bin)")) {
	end();
	cp_x_ = cp_y_ = cp_z_ = 0;
	lookat_theta_ = lookat_phi_ = .0f;
	lookat_distance_ = 10.f;
	add_object("axes", new Axes());
	add_object("target", new PointCloud(&target_, FL_YELLOW));
	model_ = new pointcloud::CylinderModel();
}

GlWindow::~GlWindow() {
	for (auto& iter : render_objects_) {
		delete iter.second;
	}
}

void GlWindow::set_center_point(float x, float y, float z) {
	cp_x_ = x; cp_y_ = y; cp_z_ = z;
	std::cout << "set center point as: (" << cp_x_ << ", " << cp_y_ << ", " << cp_z_ << ")" << std::endl;
	redraw();
}

void GlWindow::set_lookat_angle_delta(float dtheta, float dphi) {
	lookat_theta_ += dtheta; lookat_phi_ += dphi;
	std::cout << "lookat_theta: " << lookat_theta_ << ", lookat_phi: " << lookat_phi_ << std::endl;
	redraw();
}

void GlWindow::set_lookat_dis_delta(float ddis) {
	lookat_distance_ += ddis;
	if (lookat_distance_ < min_lookat_distance) {
		lookat_distance_ = min_lookat_distance;
	}
	std::cout << "lookat_distance: " << lookat_distance_ << std::endl;
	redraw();
}

int GlWindow::handle(int event) {
	static int last_x = 0, last_y = 0;
	float dtheta, dphi;
	switch (event) {
	case FL_PUSH:
		last_x = Fl::event_x();
		last_y = Fl::event_y();
		return 1;
	case FL_DRAG:
		dtheta = (Fl::event_x() - last_x) / 180.0f / x();
		dphi = (Fl::event_y() - last_y) / 180.0f / y();
		set_lookat_angle_delta(-dtheta, dphi);
		last_x = Fl::event_x();
		last_y = Fl::event_y();
		return 1;
	case FL_RELEASE:
		dtheta = (Fl::event_x() - last_x) / 180.0f / x();
		dphi = (Fl::event_y() - last_y) / 180.0f / y();
		set_lookat_angle_delta(-dtheta, dphi);
		return 1;
	case FL_MOUSEWHEEL:
		set_lookat_dis_delta(Fl::event_dy() / 5.0f);
		return 1;
	default:
		return Fl_Gl_Window::handle(event);
	}
}

void GlWindow::add_object(const char* name, RenderObject* object) {
	render_objects_[name] = object;
}

void GlWindow::remove_object(const char* name) {
	auto iter = render_objects_.find(name);
	if (iter == render_objects_.end()) return;
	delete iter->second;
	render_objects_.erase(iter);
}

void GlWindow::optimize_step(float lr) {
	if (model_ == nullptr) return;
	model_->zero_grads();
	remove_object("predicted");
	auto& predicted = model_->generate(target_.rows());
	auto loss = pointcloud::ChamferLoss(predicted, target_);
	predicted.backward(loss);
	model_->optimize(lr);
	add_object("predicted", new PointCloud(&predicted, FL_WHITE));
	redraw();
}

void GlWindow::fix_viewport(int w, int h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-w / 20000.0, w / 20000.0, -h / 20000.0, h / 20000.0, 0.1, 1000);
	set_lookat();
}

void GlWindow::resize(int x, int y, int w, int h) {
	Fl_Gl_Window::resize(x, y, w, h);
	fix_viewport(w, h);
	redraw();
}

void GlWindow::draw() {
	if (!valid()) {
		valid(1);
		fix_viewport(w(), h());
	}
	set_lookat();
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	for (auto& iter : render_objects_) {
		iter.second->render();
	}
}

void GlWindow::set_lookat() {
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(
		cp_x_ + lookat_distance_ * sin(lookat_theta_) * cos(lookat_phi_),
		cp_y_ + lookat_distance_ * sin(lookat_phi_),
		cp_z_ + lookat_distance_ * cos(lookat_theta_) * cos(lookat_phi_),
		cp_x_, cp_y_, cp_z_,
		0, 1, 0
	);
}

AppWindow::AppWindow(int w, int h, const char* l) :
	Fl_Window(w, h, l) {
	gl_window_ = new GlWindow(10, 10, w - 20, h - 80);
	for (int i = 0; i < 3; ++i) {
		center_point_text_buffer_[i] = new Fl_Text_Buffer();
		center_point_editor_[i] = new Fl_Text_Editor(10 + i * 120, h - 60, 100, 40);
		center_point_editor_[i]->textsize(20);
		center_point_editor_[i]->buffer(center_point_text_buffer_[i]);
		center_point_text_buffer_[i]->text("0");
	}
	set_center_point_btn_ = new Fl_Button(400, h - 60, 150, 40, "Set Center Point");
	set_center_point_btn_->callback(set_center_point_btn_pushed_callback_, this);
	fitting_step_btn_ = new Fl_Button(600, h - 60, 150, 40, "Fitting Step");
	fitting_step_btn_->callback(fitting_step_btn_pushed_callback_, this);
	end();
}

void AppWindow::set_center_point_btn_pushed_callback_(Fl_Widget* widget, void* data) {
	AppWindow* app_window = static_cast<AppWindow*>(data);
	float p[3];
	for (int dim = 0; dim < 3; ++dim) {
		try {
			p[dim] = std::stof(app_window->center_point_text_buffer_[dim]->text());
		}
		catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
	}
	reinterpret_cast<AppWindow*>(data)->gl_window_->set_center_point(p[0], p[1], p[2]);
}

void AppWindow::fitting_step_btn_pushed_callback_(Fl_Widget* widget, void* data) {
	AppWindow* app_window = static_cast<AppWindow*>(data);
	app_window->gl_window_->optimize_step(1.0f);
}

}