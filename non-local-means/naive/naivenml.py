import math
import numpy as np

from scipy import ndimage
from scipy import misc

# Initialize constants
noise_sigma = 10.0
patch_size = 5                  # Must be an odd number
h = 0.5
sigma = 1.0

# Generate a Gauss kernel
gauss_kernel = np.zeros((patch_size, patch_size), dtype=np.float32)

for y in xrange(-patch_size/2, patch_size/2 + 1):
    for x in xrange(-patch_size/2, patch_size/2 + 1):
        gauss_val = x*x + y*y
        gauss_val = math.exp(-gauss_val / (2.0 * sigma * sigma))
        gauss_val = gauss_val / (2.0 * math.pi * sigma * sigma)

        gauss_kernel[x + patch_size/2, y + patch_size/2] = gauss_val


def get_dist(patch1, patch2):
    mat_dist = patch1 - patch2
    mat_dist = mat_dist * mat_dist
    mat_dist = mat_dist * gauss_kernel

    return np.sum(mat_dist)


def get_patch(img, pos):
    patch = np.zeros((patch_size, patch_size), dtype=img.dtype)

    for dy in xrange(-patch_size/2, patch_size/2 + 1):
        for dx in xrange(-patch_size/2, patch_size/2 + 1):
            img_pos     = (pos[0]+dx, pos[1]+dy)
            patch_pos   = (dx + patch_size/2, dy + patch_size/2)

            if 0 <= img_pos[0] < img.shape[0] and 0 <= img_pos[1] < img.shape[1]:
                patch[patch_pos] = img[img_pos]

    return patch


def weight2D(img, pos1, pos2):
    patch1  = get_patch(img, pos1)
    patch2  = get_patch(img, pos2)
    dist    = get_dist(patch1, patch2)

    weight  = dist / (2*h*h)
    weight  = math.exp(-weight)

    return weight


def denoise2D_pixel(img, pos, verbose=False):
    img_shape = img.shape

    if verbose:
        print "Compute weights between", pos, "and remaining pixels in image..."

    weights = np.zeros(img_shape, dtype=np.float32)

    for y in xrange(img_shape[1]):
        for x in xrange(img_shape[0]):
            weights[x,y] = weight2D(img, pos, (x, y))

    if verbose:
        print "Get denoised value by averaging with weights..."

    denoised_val = 0.0
    for y in xrange(img_shape[1]):
        for x in xrange(img_shape[0]):
            denoised_val = denoised_val + weights[x,y] * img[x,y]

    if verbose:
        print "Get sum of all weights..."
    sum_weights = np.sum(weights)

    if verbose:
        print "Return normalized denoised value..."

    denoised_val = denoised_val / sum_weights
    return denoised_val


def denoise2D(img, verbose=False):
    img_shape = img.shape
    img_type  = img.dtype

    denoised_img = np.zeros(img_shape, dtype=img_type)

    for y in xrange(img_shape[1]):
        for x in xrange(img_shape[0]):
            denoised_img[x, y] = denoise2D_pixel(img, (x, y), verbose)

    return denoised_img


def main():
    print "Loading Lena image..."
    orig_img = misc.lena()[160:160+100, 160:160+100]

    print "Image dtype: %s" % orig_img.dtype
    print "Image size: %6d" % orig_img.size
    print "Image shape: %3dx%3d" % (orig_img.shape[0], orig_img.shape[1])
    print "Max value %3d at pixel %6d" % (orig_img.max(), orig_img.argmax())
    print "Min value %3d at pixel %6d" % (orig_img.min(), orig_img.argmin())
    print "Variance: %1.5f" % orig_img.var()
    print "Standard deviation: %1.5f" % orig_img.std()

    misc.imsave("orig.png", orig_img)

    print "Generating noisy image..."
    print "Noise standard deviation: %1.5f" % noise_sigma

    noise = np.random.normal(scale=noise_sigma, size=orig_img.size).reshape(orig_img.shape)
    noisy_img = orig_img + noise

    misc.imsave("noisy.png", noisy_img)

    print "Denoising image..."
    denoised_img = denoise2D(noisy_img.astype(orig_img.dtype), True)

    print "Storing denoised image..."
    misc.imsave("denoised.png", denoised_img)

if __name__ == "__main__":
    main()
