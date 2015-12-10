#include "CImg.h"
#include <time.h>
using namespace cimg_library;
typedef unsigned char uchar;

int main(){
	CImg<uchar> image("../images/donuts.bmp");
	CImg<uchar> rot(image);
	uchar * ptr;

	

	CImgDisplay main_disp(image, "Original"), draw_disp(rot, "Negatif");

	while (!main_disp.is_closed() && !draw_disp.is_closed()) {
		cimg_for(rot, ptr, uchar){ *ptr = (*ptr + 1) % 256; }
		draw_disp.display(rot);
		Sleep(20);
	}
	return 0;
}