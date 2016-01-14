from PIL import Image
import numpy as np

import optim
import img_preprocessing
from multiprocessing import Pool
import pickle
from scipy import misc


def update_dic(D, A, B):
    """
    Updates dictionary such that D minimizes 1/2 * Tr(Dt.D.A) - Tr(Dt.B)
    :param D: dictionary
    :param A: sum of alpha*alpha.T
    :param B: sum of x*alpha.T
    :return: updated dictionary
    """
    for j in range(D.shape[1]):
        u = (B[:, j] - D.dot(A[:, j]))
        u += A[j, j] * D[:, j]
        u_norm = np.sqrt(u.T.dot(u))
        if u_norm < 1e-20:
            u = np.random.rand(D.shape[0])
            u_norm = np.sqrt(u.T.dot(u))
            A[j, :] = 0.0
        u /= u_norm
        D[:, j] = u
    return D


def learn_dic(patches, dic_size, T=1000, lambd=0.1, batch_size=100):
    """
    Learns sparse coding dictionary using the online algorithm exposed in
        "J. Mairal, F. Bach, J. Ponce, and G. Sapiro. Online dictionary learning for sparse coding.
        In Proceedings of the International Conference on Machine Learning (ICML), 2009."
    :param patches: array of patches patches[i, j] represents the jth pixel of the ith patch
    :param dic_size: size of dictionary
    :param T: number of iterations
    :param lambd: Lasso regularization parameter
    :param batch_size: size of batches processed during each iteration
    :return: learned dictionary
    """

    # Extract patch size
    patch_size = patches.shape[1]

    # Initialize A and B at 0
    A = np.zeros((dic_size, dic_size))
    B = np.zeros((patch_size, dic_size))

    # Initialize D at random
    D = np.random.rand(patch_size, dic_size)

    for t in range(T):

        # Extract batch as array of column-vectors
        batch_indices = np.random.choice(range(len(patches)), size=batch_size)
        x = patches[batch_indices].T

        # Compute sparse coding of x using lars lasso
        alpha = optim.lasso_seq(D, x, lambd)

        # Update A and B
        A += alpha.dot(alpha.T) / batch_size
        B += x.dot(alpha.T) / batch_size

        # Update D
        D = update_dic(D, A, B)

        # Print percentage
        if (t + 1) % int(T / 10) == 0:
            print("\r{0}".format(int((float(t + 1) / T) * 100)), "%", end="", flush=True)

    return D


def rebuild(patch, dic, lambd):
    D = np.asarray([x.getdata() for x in dic]).transpose()
    X = img.getdata()
    alpha = optim.lasso_seq(D, X, lambd)
    X_ = D.dot(alpha)
    nimg = Image.new(img.mode, img.size)
    nimg.putdata(X_)
    nimg.save("restored.png")


def save_dic(dic, filename):
    """
    Save dictionary in textfile
    :param dic: np.array containing dictionary
    :param filename: file where the dictionary should be stored
    """
    file = open(filename, "wb+")
    pickle.dump(dic, file)
    file.close()


def load_dic(filename):
    file = open(filename, "rb")
    dic = pickle.load(file)
    file.close()
    return dic


def show_dic(dic, d):
    dic = dic.transpose()
    dic_size = dic.shape[0]
    side = np.ceil(dic_size / np.sqrt(dic_size))
    dicimg = np.zeros((side * d, side * d))
    for k, comp in enumerate(dic[:dic_size]):
        comp = comp.reshape((d, d))
        # print(np.abs(dicimgs[:, :, k] - dicimgs[:, :, k + 1]))
        i = k % side
        j = (k - i) // side
        idxi = np.arange(i * d, (i + 1) * d, dtype=int)
        idxj = np.arange(j * d, (j + 1) * d, dtype=int)
        dicimg[np.ix_(idxi, idxj)] = comp

    return dicimg


if __name__=="__main__":

    # Define parameters
    patch_size = 8  # Side length of patches
    dic_size = 100  # Size of dictionary (number of atoms)
    regularization = 1.2 / dic_size  # Regularization parameter of the Lasso optimization problem
    batch_size = 3  # Size of batches
    n_iter = 1000  # Number of iterations

    # Image name
    img_filename = "../images/donuts.png"

    # Open image
    img = Image.open(img_filename)

    # Get patches from image
    patches = img_preprocessing.get_patches(img, patch_size)

    # Learn dictionary
    D = learn_dic(patches, dic_size, T=n_iter, batch_size=batch_size, lambd=regularization)

    # Save dictionary to file
    save_dic(D, "dic.txt")

    # Save dictionary as nice image
    dicimg = show_dic(D, patch_size)
    misc.imsave("dicimg.jpg", dicimg)
