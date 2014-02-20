#!/usr/bin/env python

import math
import numpy as np
import random
import scipy.io as sio
import sys
import os.path

from scipy import ndimage
from scipy import misc
from scipy.cluster import vq

# Initialize constants
num_clusters = 5
max_iters = 10
patch_radius = 2                 # Neighborhood size


def circshift3d(arr, shift):
    shift_d, shift_h, shift_w = shift
    depth, height, width = arr.shape

    shift_d = np.sign(shift_d) * (abs(shift_d) % depth)
    shift_h = np.sign(shift_h) * (abs(shift_h) % height)
    shift_w = np.sign(shift_w) * (abs(shift_w) % width)

    arr = np.tile(arr, (2, 2, 2))

    ini_d = (depth - shift_d) % depth
    ini_h = (height - shift_h) % height
    ini_w = (width - shift_w) % width

    end_d = ini_d + depth
    end_h = ini_h + height
    end_w = ini_w + width

    return arr[ini_d:end_d, ini_h:end_h, ini_w:end_w]


def build_patches(vol, patch_radius):
    depth, height, width = vol.shape
    patch_size = (2*patch_radius + 1) ** 3

    patches = np.zeros((patch_size, depth, height, width))
    n = 0

    for k in range(patch_radius, -patch_radius-1, -1):
        for j in range(patch_radius, -patch_radius-1, -1):
            for i in range(patch_radius, -patch_radius-1, -1):
                shift_img = circshift3d(vol, (k, j, i))
                patches[n, :, :, :] = shift_img
                n += 1

    return patches


def load_input_vol():
    file_name = sys.argv[1]
    vol_name = os.path.splitext(file_name)[0]

    print "Loading input volume %s..." % vol_name
    file_contents = sio.loadmat(file_name)

    for key in file_contents:
        value = file_contents[key]
        if type(value) == np.ndarray:
            vol = value

    return vol, vol_name


def save_output_vol(vol, vol_name):
    file_name = sys.argv[2]
    sio.savemat(file_name, {vol_name: vol})


def main():
    if len(sys.argv) != 3:
        print "Usage: %s %s %s" % (sys.argv[0], "<input.mat>", "<ouput.mat>")
        sys.exit(-1)

    # Load volume and print stats
    vol, vol_name = load_input_vol()
    depth, height, width = vol.shape

    print "Volume dtype: %s" % vol.dtype
    print "Volume size: %6d" % vol.size
    print "Volume shape: %3dx%3dx%3d" % (vol.shape[0], vol.shape[1], vol.shape[2])
    print "Max value %3d at voxel %6d" % (vol.max(), vol.argmax())
    print "Min value %3d at voxel %6d" % (vol.min(), vol.argmin())
    print "Variance: %1.5f" % vol.var()
    print "Standard deviation: %1.5f" % vol.std()

    # Generate patches for each voxel
    print "Generating patches..."
    patch_size = (2*patch_radius + 1) ** 3
    patches = build_patches(vol, patch_radius)
    patches = patches.reshape(patch_size, depth*height*width)
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

    # print "Saving image..."
    clusters = clusters.reshape((depth, height, width))
    clusters = clusters.astype(vol.dtype)
    save_output_vol(clusters, vol_name)

if __name__ == "__main__":
    main()
