#include "CImg.h"
#include <time.h>
#include <iostream>

using namespace std;

using namespace cimg_library;
typedef unsigned char uchar;

#define ANTOINE

#ifdef ANTOINE
#define FOLDER "/Users/antoineprouvost/GitHub/sparse_coding/images/"
#endif

#ifdef PAUL
#define FOLDER "../images/"
#endif

#define FILE "donuts.bmp"


int main(){

	CImg<uchar> image(FOLDER FILE);
	CImg<uchar> rot(image);
	uchar * ptr;

	

	CImgDisplay main_disp(image, "Original"), draw_disp(rot, "Negatif");

	while (!main_disp.is_closed() && !draw_disp.is_closed()) {
		cimg_for(rot, ptr, uchar){ *ptr = (*ptr + 1) % 256; }
		draw_disp.display(rot);
//		sleep(20);
	}

	return 0;
}