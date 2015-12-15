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
    void restore();

private:
	CImg<uchar> img;
    CImgDisplay disp;
    int patch_size;
	CImgList<uchar> patches;
    int dic_size;
    CImgList<uchar> dictionary;
    double lambda; //regularisation parameter for dictionary learning
    CImgList<uchar> A;
    CImgList<uchar> B;


	void build_patches();
    
    CImg<uchar> LARS(int t);
    
    void dic_update(int t);
    
    void dic_learn();
    

};

CImgList<uchar> dot(CImgList<uchar> X, CImgList<uchar> Y);