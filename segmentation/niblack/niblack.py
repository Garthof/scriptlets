"""
Provides several functions to segment a 2D or 3D array using Niblack's
segmentation method. This is a local thresholding algorithm where the
threshold of each point x is given by:

t(x) = m_r(x) + k * s_r(x)

m_r(x) is the mean value of a neighborhood surrounding x with radius r; s_r(x)
is the corresponding standard deviation of the same neighborhood. r and k are
paramenters of the algorithm.
"""

import numpy as np
import matplotlib.pyplot.imshow

def __mask_segm(data, mask):
    if mask is not None:
        data += 1
        data *= (mask > 0)
    return data


def thres_2d(img, r, k):
    """
    Computes the local threshold for each point in the array. A 2D neighborhood
    of radius r is used.
    """
    patch_size = (2*r + 1)**2
    patches = np.zeros((img.shape[0], img.shape[1], patch_size), dtype=img.dtype)
    count = 0

    for x in xrange(-r, r+1):
        for y in xrange(-r, r+1):
            patches[:, :, count] = np.roll(np.roll(img, x, axis=0), y, axis=1)
            count += 1

    threshold = np.mean(patches, axis=2) + k * np.std(patches, axis=2)
    return threshold


def thres_layer(vol, r, k):
    """
    Computes the local threshold for each point in the array. Each layer is
    examined individually using a 2D neighborhood of radius r.
    """
    threshold = np.zeros(vol.shape)

    for z in range(vol.shape[2]):
        threshold[:, :, z] = thres_2d(vol[:, :, z], r, k)

    return threshold


def thres_3d(vol, r, k):
    """
    Computes the local threshold for each point in the array. A 3D neighborhood
    of radius r is used.
    """

    patch_size = (2*r + 1)**3
    patches = np.zeros((vol.shape[0], vol.shape[1], vol.shape[2], patch_size),
                       dtype=vol.dtype)
    count = 0

    for x in xrange(-r, r+1):
        for y in xrange(-r, r+1):
            for z in xrange(-r, r+1):
                patches[:, :, :, count] = np.roll(np.roll(np.roll(vol, x, axis=0), y, axis=1), z, axis=2)
                count += 1

    threshold = np.mean(patches, axis=3) + k * np.std(patches, axis=3)
    return threshold


def segm_2d(img, r, k, mask=None):
    """
    Returns a segmented dataset using Niblack's method. The mask is applied
    after the segmentation.
    """
    segm_img = (img > thres_2d(img, r, k)).astype(int)
    return __mask_segm(segm_img, mask)


def segm_layer(vol, r, k, mask=None):
    """
    Returns a segmented dataset using Niblack's method. The mask is applied
    after the segmentation.
    """
    segm_vol = (vol > thres_layer(vol, r, k)).astype(int)
    return __mask_segm(segm_vol, mask)


def segm_3d(vol, r, k, mask=None):
    """
    Returns a segmented dataset using Niblack's method. The mask is applied
    after the segmentation.
    """
    segm_vol = (vol > thres_3d(vol, r, k)).astype(int)
    return __mask_segm(segm_vol, mask)


def show_2d(img):
    """ Shows a 2D gray image. """
    imshow(img, cmap=cm.gray, interpolation='none')


def show_3d(vol, slice=None):
    """ Shows a slice of a 3D volume as a gray image. """
    if slice is None: slice = vol.shape[2] // 2
    imshow(vol[:, :, slice], cmap=cm.gray, interpolation='none')