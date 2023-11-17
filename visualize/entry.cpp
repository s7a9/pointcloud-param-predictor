#include <iostream>

#include "window.h"

int main(int argc, char* argv[]) {
	using namespace visual;
	AppWindow win(1200, 800, "visualize");
	win.resizable(win);
	win.show();
	return Fl::run();
}
