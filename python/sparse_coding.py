from PIL import Image
import numpy as np
import cvxpy as cp
from scipy import misc
from sklearn.linear_model import LassoLars
from random import uniform, seed
from datetime import datetime
import pickle
from math import sqrt


def get_patches(img, size):
    (c, r) = img.size
    assert c >= size and r >= size
    ret = np.array(
            [misc.fromimage(img.crop((x, y, x + size, y + size)), flatten=True).flatten() for x in range(c - size) for
             y in range(r - size)]).T
    std = np.std(ret, axis=1)[:, np.newaxis]
    mean = np.mean(ret, axis=1)[:, np.newaxis]
    ret = (ret - mean) / (0.1 + std)
    return ret


def update_dic(D, A, B):
    """

    :param D: dictionnary
    :param A: sum of alpha*alpha.T
    :param B: sum of x*alpha.T
    :return:
    """
    ret = D.copy()
    prec = np.ones(ret.shape) * np.Inf
    while np.linalg.norm(prec - ret) >= 0.0001:
        prec = ret.copy()
        for j in range(D.shape[1]):
            u = (B[:, j] - D.dot(A[:, j])) / A[j, j] + D[:, j]
            ret[:, j] = u / max((np.linalg.norm(u), 1))
    return ret


def lasso(D, x, gamma):
    """
    Computes Lasso optimization
    :param D: dictionnary
    :type D: np.array
    :param x: image
    :type x: np.array
    :param gamma: regularization factor
    :type gamma: float
    :rtype: np.array
    """
    num_samples = x.shape[1]
    alpha = cp.Variable(D.shape[1], x.shape[1])
    objective = cp.Minimize(sum(cp.square(D * alpha - x)) / num_samples + gamma * cp.norm(alpha, 1))

    P = cp.Problem(objective)
    P.solve()
    return alpha.value


def learn_dic(img_list, dic_size, T=1000, lambd=0.1, kappa=0.01, batch_size=100):
    """

    :param img_list: list(Image)
    :param k: int
    :param T: int
    """
    seed(datetime.now())
    num_images = img_list.shape[1]
    img_size = img_list.shape[0]
    A = np.eye(dic_size, dic_size) * kappa
    B = np.zeros((img_size, dic_size))
    D = np.zeros((img_size, dic_size))  # np.random.rand(img_size, dic_size)
    for t in range(T):
        i = np.random.choice(range(num_images), size=batch_size)
        x = img_list[:, i]
        alpha = lasso(D, x, lambd)
        A += alpha.dot(alpha.T) / batch_size
        B += B + x.dot(alpha.T) / batch_size
        D = update_dic(D, A, B)

        if (t + 1) % int(T / 10) == 0:
            print("\r{0}".format(int((float(t + 1) / T) * 100)), "%", end="", flush=True)

    return D


def rebuild(img, dic, lambd):
    D = np.asarray([x.getdata() for x in dic]).transpose()
    X = img.getdata()
    LARS = LassoLars(alpha=lambd / img.size[0])
    LARS.fit(D, X)
    alpha = LARS.coef_
    X_ = D.dot(alpha)
    nimg = Image.new(img.mode, img.size)
    nimg.putdata(X_)
    nimg.save("restored.png")


def rebuild_img(patches, dic, lambd, img):
    restored_patches = []
    (c, r) = img.size
    size = patches[0].size[0]
    for p in patches:
        D = np.asarray([x.getdata() for x in dic]).transpose()
        X = p.getdata()
        LARS = LassoLars(alpha=lambd / p.size[0])
        LARS.fit(D, X)
        alpha = LARS.coef_
        X_ = D.dot(alpha)
        nimg = Image.new(p.mode, p.size)
        nimg.putdata(X_)
        restored_patches.append(nimg)
    restored_patches = [[restored_patches[y + (r - size) * x] for x in range(c - size)] for y in range(r - size)]
    restored_img = Image.new(patches[0].mode, (c, r))
    for x in range(c - size):
        for y in range(r - size):
            for dx in range(size):
                for dy in range(size):
                    v = restored_img.getpixel((x + dx, y + dy))
                    restored_img.putpixel((x + dx, y + dy), v + restored_patches[y][x].getpixel((dx, dy)))
    for x in range(c):
        for y in range(r):
            v = restored_img.getpixel((x, y))
            norm = min((min(x + 1, size), min(c - x, size))) * min((min(y + 1, size), min(r - y, size)))
            restored_img.putpixel((x, y), v / norm)
    return restored_img


def show_dic(dic, d):
    dic_size = dic.shape[1]
    side = np.ceil(dic_size / sqrt(dic_size))
    dicimgs = dic.reshape((d, d, -1))
    assert (dicimgs[:, :, 10].flatten() == dic[:, 10]).all()
    dicimg = np.zeros((side * d, side * d))
    for k in range(dic_size):
        # print(np.abs(dicimgs[:, :, k] - dicimgs[:, :, k + 1]))
        i = k % side
        j = (k - i) // side
        idxi = np.arange(i * d, (i + 1) * d, dtype=int)
        idxj = np.arange(j * d, (j + 1) * d, dtype=int)
        dicimg[np.ix_(idxi, idxj)] = dicimgs[:, :, k]

    return dicimg


patch_size = 8
img = Image.open("../images/donuts.png")
img = img.convert("L")

patches = get_patches(img, 8)
m = 100
D = learn_dic(patches, m, T=100, batch_size=100, lambd=1.2 / sqrt(m))
std = np.std(D,axis=1)
print(std.shape)
print(std)
with open("dic.txt", "wb+") as f:
    pickle.dump(D, f)
with open("dic.txt", "rb") as f:
    D = pickle.load(f)
    dicimg = show_dic(D, patch_size)
    misc.imsave("dicimg.jpg", dicimg)
    # rebuild_img(patches, D, 1.2 / sqrt(m), img).save("restored_img.png")
    # rebuild(patches[10], D, 1.2 / sqrt(m))
    # patches[10].save("original.png")
