#include "sparse_coding.h"

void print(arma::mat m){
    for(int i=0; i< m.n_rows; i++){
        cout << m.row(i) << endl;
    }
    
}



// SparseCoding functions

SparseCoding::SparseCoding(const char* filename, int patch_size, int dic_size, double lambda){
	this->patch_size = patch_size;
    this->dic_size = dic_size;
    this->lambda = lambda;
    
	CImg<double> buff_img = CImg<double>(filename);
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
	CImg<double> n_img(size * patch_size, size * patch_size);
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

void SparseCoding::showDic(){
    int size = static_cast<int>( sqrt(dic_size) ) + 1;
    CImg<double> n_img(size * patch_size, size * patch_size);
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            for (int x = 0; x < patch_size; x++){
                for (int y = 0; y < patch_size; y++){
                    n_img(i * patch_size + x, j * patch_size + y)
                    = i+size*j< dic_size ? dictionary.col(i+size*j)(x + y*patch_size) : 0;
                }
            }
            
        }
    }
    disp.display(n_img);
    while (! disp.wait().is_keyENTER());
}

void SparseCoding::build_patches(){
	patches = CImgList<double>((img.width() - patch_size)*(img.height() - patch_size), patch_size, patch_size);
	for (int x = 0; x < img.width() - patch_size; x++){
		for (int y = 0; y < img.height() - patch_size; y++){
			cimg_forXY(patches[x*(img.height() - patch_size) + y], i, j){
				patches(x*(img.height() - patch_size) + y, i, j) = img(x + i, y + j);
			}
		}
	}
}

arma::vec SparseCoding::LARS(){
    //LARS object to compute algorithm
    mlpack::regression::LARS L(true, lambda, 0.0);
    
    //solution vector of alpha = argmin 1/2|| x - dictionary * alpha||_2 ^2 + lamda ||alpha|_1
    arma::vec alpha(dictionary.n_cols);
    
    //The input matrix (like all mlpack matrices) should be column-major
    //each column is an observation and each row is a dimension.
    
    //mlpack 1.0.12 version
    L.Regress(dictionary, x, alpha, false);
    
    //mlpack 2.0.0 version
    //L.Train(dictionary, x, alpha, false);
    
    return alpha;
}

void SparseCoding::dic_update(){
    //temporary vector u
    arma::vec u;
    //jth column of dictionary being updated
    for(int j=0; j<dictionary.n_cols; j++){
        u =  (B.col(j) - dictionary * A.col(j));
        if(A(j,j) !=0 )
            u =  u / A(j,j);
        u += dictionary.col(j);
        u = arma::norm(u)  > 1 ? u/arma::norm(u) : u;
        for(int i=0; i< dictionary.n_rows; i++)
            dictionary(i,j) = u(i);
    }
}

void SparseCoding::dic_learn(int T){
    //initiate dictionnary
    dictionary = arma::mat(patch_size*patch_size, dic_size);

    srand(time(0));
    for(int j=0; j< dic_size; j++){
        int n_patch = patches.size() * ((double)rand() / RAND_MAX);
        arma::vec x = patchTovec(patches(n_patch));
        for(int i=0; i< dictionary.n_rows; i++)
            dictionary(i,j) = x(i)/255;
    }
        showDic();
    //initiate x and alpha
    arma::vec alpha;
    
    //initiate A and B
    A = arma::mat(dictionary.n_cols, dictionary.n_cols);
    A.zeros();
    B = arma::mat(dictionary.n_rows, dictionary.n_cols);
    B.zeros();
    
    //iterate learning
    for (int t=1; t<=T; t++) {
        //draw x a patch
        x=patchTovec(patches(patches.size() * ((double)rand() / RAND_MAX)));
        
        //LARS to compute alpha = argmin 1/2|| x - dictionary * alpha||_2 ^2 + lamda ||alpha|_1
        alpha = LARS();
        
        //update two matrices A and B
        A = A + alpha * alpha.t();
        B = B + x * alpha.t();
        //update dictionary using A and B
        dic_update();
    }
}

arma::vec SparseCoding::patchTovec(CImg<double> I){
    arma::vec u(I.height()*I.width() );
    for(int x=0; x<I.height(); x++)
        for (int y=0; y<I.width(); y++)
            u(x+y*I.height()) = I(x,y);
    return u;
}

CImg<double> SparseCoding::vecTopatch(arma::vec u){
    CImg<double> I(patch_size, patch_size);
    cimg_forXY(I, x, y){
        I(x,y) = u(x+y*I.height());
    }
    return I;
}

void SparseCoding::restore(){
    dic_learn(100);
}


