#include "sparse_coding.h"
#include <time.h>
#include <iostream>
#include <stdlib.h>


#define ANTOINE

#ifdef ANTOINE
#define FOLDER "/Users/antoine/GitHub/sparse_coding/images/"
#endif

#ifdef PAUL
#define FOLDER "../images/"
#endif

#define FILE "para_noise.bmp"

using namespace std;

int main(){

    int patch_size = 5;
	SparseCoding sc(FOLDER FILE, patch_size, 256, 1.2/patch_size);

	sc.showImage();
//
//	sc.showRandomPatches();
//	sc.showRandomPatches();
    sc.showRandomPatches();
//    
    sc.restore();
    sc.showDic();
}