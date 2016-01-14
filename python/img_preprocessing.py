from PIL import Image
import numpy as np
from scipy.misc import fromimage
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


def get_patches(img, patch_size):
    img_array = fromimage(img, flatten=True)
    patches = extract_patches_2d(img_array, (patch_size, patch_size))
    patches = patches.reshape(patches.shape[0], -1)
    patches = center(patches)
    return patches


def center(x, axis=0):
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    return (x - mean) / std
