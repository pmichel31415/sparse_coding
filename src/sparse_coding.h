#include "CImg.h"
#include "time.h"
#include <iostream>
#include <vector>
#include <mlpack/methods/lars/lars.hpp>

using namespace std;
using namespace cimg_library;

class SparseCoding
{
public:
    
    SparseCoding(const char* filename, int patch_size, int dic_size, double lambda);

	void showImage();
	void showRandomPatches();
    void showDic();
    void restore();

    
private:
    
	CImg<double> img; //original image
    CImgDisplay disp; //GUI windoe to show images
    
    int patch_size;
	CImgList<double> patches; //list of all available patches in the original image
    
    int dic_size; //number of patches considered in the dictionary
    arma::mat dictionary; //each column is a patch and each row is a dimension.
    double lambda; //regularisation parameter for dictionary learning
    
    arma::mat A; //temporary matrix used in algo
    arma::mat B; //temporary matrix used in algo
    arma::vec x; //temporary patch used in algo


	void build_patches();
    
    arma::vec LARS();
    
    void dic_update();
    
    void dic_learn(int T);
    
    arma::vec patchTovec(CImg<double> I);
    
    CImg<double> vecTopatch(arma::vec x);
    

};