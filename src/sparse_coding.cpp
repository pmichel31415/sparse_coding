#include "sparse_coding.h"

// SparseCoding functions

SparseCoding::SparseCoding(const char* filename, int p)
{
	patch_size = p;
	CImg<uchar> buff_img = CImg<uchar>(filename);
	// Color to B&W
	img = (buff_img.spectrum() > 1) ? buff_img.get_RGBtoYCbCr().channel(0) : buff_img;

	build_patches();
}


void SparseCoding::showImage(){
	disp.display(img);
	while (! disp.wait().is_keyENTER());
}

void SparseCoding::showRandomPatches(){
	srand(time(0));
	int size = 30;
	CImg<uchar> n_img(size * patch_size, size * patch_size);
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			unsigned int n_patch = patches.size() * ((double)rand() / RAND_MAX);
			for (int x = 0; x < patch_size; x++){
				for (int y = 0; y < patch_size; y++){
					n_img(i * patch_size + x, j * patch_size + y) = patches(n_patch, x, y);
				}
			}


		}
	}
	disp.display(n_img);
	while (! disp.wait().is_keyENTER());
}

void SparseCoding::build_patches(){
	patches = CImgList<uchar>((img.width() - patch_size)*(img.height() - patch_size), patch_size, patch_size);
	for (int x = 0; x < img.width() - patch_size; x++){
		for (int y = 0; y < img.height() - patch_size; y++){
			cimg_forXY(patches[x*(img.height() - patch_size) + y], i, j){
				patches(x*(img.height() - patch_size) + y, i, j) = img(x + i, y + j);
			}
		}
	}
}

CImg<uchar> SparseCoding::LARS(int t){}

void SparseCoding::dic_update(int t){}

void SparseCoding::dic_learn(){}

void SparseCoding::restore(){}
