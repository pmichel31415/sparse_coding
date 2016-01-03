#include "sparse_coding.h"
#include <time.h>
#include <iostream>


#define ANTOINE

#ifdef ANTOINE
#define FOLDER "/Users/antoineprouvost/GitHub/sparse_coding/images/"
#endif

#ifdef PAUL
#define FOLDER "../images/"
#endif

#define FILE "donuts.bmp"


int main(){

	SparseCoding sc(FOLDER FILE, 5, 10, 0.5);

	sc.showImage();

//	sc.showRandomPatches();
//	sc.showRandomPatches();
	sc.showRandomPatches();
    
    sc.restore();
}