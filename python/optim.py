import numpy as np
import cvxpy as cp
from scipy import misc, linalg
from sklearn.linear_model import LassoLars
from sklearn.decomposition import MiniBatchDictionaryLearning


def lasso_sklearn(dict, target, gamma):
    """
    Computes Lasso optimization
    :param dict: dictionnary
    :type dict: np.array
    :param target: image
    :type target: np.array
    :param gamma: regularization factor
    :type gamma: float
    :rtype: np.array
    """

    num_samples = target.shape[1]
    patch_size = dict.shape[0]
    dic_size = dict.shape[1]
    gamma /= num_samples
    ll = LassoLars(alpha=gamma, fit_intercept=False, normalize=False, fit_path=False)
    ll.fit(dict, target)

    alpha = ll.coef_

    alpha = alpha.reshape(dic_size, num_samples)
    return alpha


def lasso_cvxpy(dict, target, gamma):
    """
    Computes Lasso optimization
    :param dict: dictionnary
    :type dict: np.array
    :param target: image
    :typetarget: np.array
    :param gamma: regularization factor
    :type gamma: float
    :rtype: np.array
    """
    num_samples = target.shape[1]
    patch_size = dict.shape[0]
    dic_size = dict.shape[1]
    alpha = cp.Variable(dic_size, num_samples)
    D = cp.Parameter(patch_size, dic_size, value=dict)
    x = cp.Parameter(patch_size, num_samples, value=target)
    objective = cp.Minimize(sum(cp.square(D * alpha - x)) / (2 * num_samples) + gamma * cp.norm(alpha, 1))

    P = cp.Problem(objective)
    P.solve()
    # print(P.value)
    return alpha.value


def lasso_function(x):
    return lasso_sklearn(*x)


def lasso(dict, target, gamma):
    batch_size = target.shape[1]
    pool = Pool()
    args = [(dict.copy(), target[:, k][:, np.newaxis].copy(), gamma) for k in range(batch_size)]

    res = pool.map(lasso_function, args)
    alphas = np.zeros((dict.shape[1], batch_size))
    for i, alpha in enumerate(res):
        alphas[:, i] = res[i][:, 0]
    pool.close()
    pool.join()
    return alphas


def lasso_seq(dict, target, gamma):
    batch_size = target.shape[1]
    alphas = np.zeros((dict.shape[1], batch_size))
    for i in range(batch_size):
        alphas[:, i] = lasso_sklearn(dict, target[:, i][:, np.newaxis], gamma)[:, 0]
    return alphas


def sklearn_check(img, patch_size, dic_size, T=1000):
    patch_shape = (patch_size, patch_size)
    patches = extract_patches_2d(img, patch_shape)
    patches = patches.reshape(patches.shape[0], -1)
    patches = center(patches)
    dl = MiniBatchDictionaryLearning(dic_size, n_iter=T)
    dl.fit(patches)
    D = dl.components_.T
    return D
