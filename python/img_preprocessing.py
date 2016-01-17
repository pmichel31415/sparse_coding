from PIL import Image
import numpy as np
from scipy.misc import fromimage, toimage
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


def get_patches(img, patch_size):
    img_array = fromimage(img, flatten=True)
    patches = extract_patches_2d(img_array, (patch_size, patch_size))
    patches = patches.reshape((patches.shape[0], -1))
    patches = center(patches)
    return patches


def center(x, axis=0):
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return (x - mean) / std


def build_from_patches(patches, img_size):
    patch_size = int(np.sqrt(patches.shape[1]))
    assert patch_size ** 2 == patches.shape[1], "Non square patch size"
    patches = patches.reshape((patches.shape[0], patch_size, patch_size))
    img = toimage(reconstruct_from_patches_2d(patches, (img_size[1],img_size[0])))
    return img
