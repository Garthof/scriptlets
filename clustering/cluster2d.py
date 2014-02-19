#!\usr\bin\env python

import math
import numpy as np
import random
import sys
import os.path

from scipy import ndimage
from scipy import misc

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
        return misc.lena()[160:160+img_size[0], 160:160+img_size[1]]
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


def init_means(patches, num_clusters):
    patch_size, height, width = patches.shape

    # Select random positions in the data
    rand_indices = range(height*width)
    random.shuffle(rand_indices)
    rand_indices = rand_indices[:num_clusters]

    # Get patches in those positions
    means = np.zeros((num_clusters, patch_size))
    patches = patches.reshape(patch_size, height*width)
    patches = patches.transpose()

    for k, rand_index in enumerate(rand_indices):
        means[k, :] = patches[rand_index, :]

    return means


def compute_clusters(patches, means):
    patch_size, height, width = patches.shape
    num_clusters = means.shape[0]

    # Get each patch as a row
    patches = patches.reshape(patch_size, height*width)
    patches = patches.transpose()

    # Compute distances to means
    dists = np.zeros((height*width, num_clusters))

    for i in range(num_clusters):
        dists[:, i] = np.sum((patches - means[i])**2, axis=1)

    # Get which mean is the minimum for each patch. The id of the mean is the
    # id of the cluster
    clusters = np.argmin(dists, axis=1)
    return clusters.reshape((height, width))


def compute_means(patches, clusters):
    patch_size, height, width = patches.shape

    # Get each patch as a row
    patches = patches.reshape(patch_size, height*width)
    patches = patches.transpose()

    # Compute means from current clusters
    means = np.zeros((height*width, patch_size))

    for k in range(height*width):
        # Build a mask to hide patches that do not belong to the cluster
        mask = np.tile((clusters == k).reshape(height*width, 1), patch_size)

        # Use mask to build masked array of patches
        masked_patches = np.ma.masked_array(patches, mask=~mask)

        # Compute new mean for this patch
        means[k, :] = np.mean(masked_patches, axis=0).data

    return means


def main():
    # Load image and print stats
    orig_img = get_input_img()

    print "Image dtype: %s" % orig_img.dtype
    print "Image size: %6d" % orig_img.size
    print "Image shape: %3dx%3d" % (orig_img.shape[0], orig_img.shape[1])
    print "Max value %3d at pixel %6d" % (orig_img.max(), orig_img.argmax())
    print "Min value %3d at pixel %6d" % (orig_img.min(), orig_img.argmin())
    print "Variance: %1.5f" % orig_img.var()
    print "Standard deviation: %1.5f" % orig_img.std()

    misc.imsave(img_name + "-input.png", orig_img)

    # Generate patches for each pixel
    print "Initialize clusters..."
    patches = build_patches(orig_img, patch_radius)
    means = init_means(patches, num_clusters)

    print "Refine clusters..."
    for i in range(max_iters):
        print "Iteration %d" % i
        old_clusters = None

        # Compute new clusters. If the new clusters are different from the old
        # ones, we have reached an optimum. Otherwise, new means are recomputed
        # and the iterative process continues
        clusters = compute_clusters(patches, means)

        if np.all(old_clusters == clusters):
            break
        else:
            means = compute_means(patches, clusters)
            old_clusters = clusters.copy()

    print "Saving image..."
    misc.imsave("clusters.png", 255. * clusters / (num_clusters * 1.))

if __name__ == "__main__":
    main()
