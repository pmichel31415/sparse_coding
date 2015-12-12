#include "CImg.h"
#include "time.h"
#include <iostream>
#include <vector>

using namespace std;
using namespace cimg_library;

typedef unsigned char uchar;
typedef vector<uchar> vec;
typedef vector<vec> mat;

class SparseCoding
{
public:
	SparseCoding(const char* filename,int s=5);

	void showImage();
	void showRandomPatches();

private:
	int patch_size;
	CImg<uchar> img;
	CImgList<uchar> patches;
	CImgList<uchar> D;
	CImgDisplay disp;

	void build_patches();

};