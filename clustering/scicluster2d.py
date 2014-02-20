#!\usr\bin\env python

import math
import numpy as np
import random
import sys
import os.path

from scipy import ndimage
from scipy import misc
from scipy.cluster import vq

# Initialize constants
num_clusters = 5
max_iters = 10
patch_radius = 2                 # Neighborhood size


def circshift2d(arr, shift):
    shift_h, shift_w = shift
    height, width = arr.shape

    shift_h = np.sign(shift_h) * (abs(shift_h) % height)
    shift_w = np.sign(shift_w) * (abs(shift_w) % width)

    arr = np.tile(arr, (2, 2))

    ini_h = (height - shift_h) % height
    ini_w = (width - shift_w) % width

    end_h = ini_h + height
    end_w = ini_w + width

    return arr[ini_h:end_h, ini_w:end_w]


def get_input_img():
    global img_name

    if len(sys.argv) == 1:
        img_name = "lena"
        img_size = 50, 50

        print "Using default Lena image..."
        return misc.lena()
    else:
        file_name = sys.argv[1]
        img_name = os.path.splitext(file_name)[0]

        print "Loading input image %s..." % img_name
        return misc.imread(file_name)


def build_patches(img, patch_radius):
    height, width = img.shape
    patch_size = (2*patch_radius + 1) ** 2

    patches = np.zeros((patch_size, height, width))
    k = 0

    for j in range(patch_radius, -patch_radius-1, -1):
        for i in range(patch_radius, -patch_radius-1, -1):
            shift_img = circshift2d(img, (j, i))
            patches[k, :, :] = shift_img
            k += 1

    return patches


def main():
    # Load image and print stats
    img = get_input_img()
    height, width = img.shape

    print "Image dtype: %s" % img.dtype
    print "Image size: %6d" % img.size
    print "Image shape: %3dx%3d" % (img.shape[0], img.shape[1])
    print "Max value %3d at pixel %6d" % (img.max(), img.argmax())
    print "Min value %3d at pixel %6d" % (img.min(), img.argmin())
    print "Variance: %1.5f" % img.var()
    print "Standard deviation: %1.5f" % img.std()

    misc.imsave(img_name + "-input.png", img)

    # Generate patches for each pixel
    print "Generating patches..."
    patch_size = (2*patch_radius + 1) ** 2
    patches = build_patches(img, patch_radius)
    patches = patches.reshape(patch_size, height*width)
    patches = patches.transpose()

    # Normalize patches (standard procedure when doing k-means), and compute
    # k-means
    norm_patches = vq.whiten(patches)

    # Based on http://stackoverflow.com/a/20661301/1679, and the fact that
    # kmeans2 returns the clusters, I use this function instead of kmeans.
    # For some reason, the "random" init mode fails in my computer
    print "Computing k-means..."
    means, clusters = vq.kmeans2(norm_patches, num_clusters,
                                 iter=max_iters, minit="points")

    print "Saving image..."
    clusters = clusters.reshape((height, width))
    misc.imsave("clusters.png", 255. * clusters / (num_clusters * 1.))

if __name__ == "__main__":
    main()
