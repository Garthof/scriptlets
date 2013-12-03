import math
import numpy as np

from scipy import ndimage
from scipy import misc

# Initialize constants
noise_sigma = 10.0
window_size = 21                # Search window size - must be an odd number
patch_size = 5                  # Neighborhood size - must be an odd number
h = 0.25
sigma = 1.0

# Generate a Gauss kernel
gauss_kernel = np.zeros((patch_size, patch_size), dtype=np.float32)

for y in xrange(-patch_size/2, patch_size/2 + 1):
    for x in xrange(-patch_size/2, patch_size/2 + 1):
        gauss_val = x*x + y*y
        gauss_val = math.exp(-gauss_val / (2.0 * sigma * sigma))
        gauss_val = gauss_val / (2.0 * math.pi * sigma * sigma)

        gauss_kernel[x + patch_size/2, y + patch_size/2] = gauss_val

def is_within(pos, shape):
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]


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

            if is_within(img_pos, img.shape):
                patch[patch_pos] = img[img_pos]

    return patch


def weight2D(img, pos1, pos2):
    # The weight between equal positions should be one, as the distance is zero.
    # However, this is a special case, and weight is computed differently. See
    # more information in denoise2D_pixel
    if pos1 == pos2:
        return 0.

    # Get neighboring patches for the two positions and compute distances
    patch1  = get_patch(img, pos1)
    patch2  = get_patch(img, pos2)
    dist    = get_dist(patch1, patch2)

    weight  = dist / (2*h*h)
    weight  = math.exp(-weight)

    return weight


def denoise2D_pixel(img, img_pix_pos, verbose=False):
    """
        Returns the value of a single pixel after applying denoising.
        @param img image to be denoised
        @param img_pix_pos position of the pixel in the image
    """
    if verbose:
        print "Compute denoised value for pixel in", img_pix_pos, "..."

    win_shape       = (window_size, window_size)
    win_weights     = np.zeros(win_shape, dtype=np.float32)
    denoised_val    = 0.0

    for dy in xrange(-window_size/2, window_size/2 + 1):
        for dx in xrange(-window_size/2, window_size/2 + 1):
            # Neighbor's position in the image
            img_ngh_pos = (img_pix_pos[0] + dx, img_pix_pos[1] + dy)

            # Neighbor's position in the window
            win_ngh_pos = (dx + window_size/2, dy + window_size/2)

            # Get weight for current neighbor
            if is_within(img_ngh_pos, img.shape):
                win_weights[win_ngh_pos] = weight2D(img, img_pix_pos, img_ngh_pos)
                denoised_val += win_weights[win_ngh_pos] * img[img_ngh_pos]

    # Add own contribution. This is a special case, and contribution from its
    # own pixel is accounted by using the maximum weight of all neighbouring
    # pixels within the window
    max_weights = np.max(win_weights)
    denoised_val += max_weights * img[img_pix_pos]

    # Sum all weights to normalize all weight contributions
    sum_weights = np.sum(win_weights) + max_weights
    denoised_val /= sum_weights

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
    img_size = 50, 50
    orig_img = misc.lena()[160:160+img_size[0], 160:160+img_size[1]]

    print "Image dtype: %s" % orig_img.dtype
    print "Image size: %6d" % orig_img.size
    print "Image shape: %3dx%3d" % (orig_img.shape[0], orig_img.shape[1])
    print "Max value %3d at pixel %6d" % (orig_img.max(), orig_img.argmax())
    print "Min value %3d at pixel %6d" % (orig_img.min(), orig_img.argmin())
    print "Variance: %1.5f" % orig_img.var()
    print "Standard deviation: %1.5f" % orig_img.std()

    misc.imsave("orig.png", orig_img)

    # Generate additive white Gaussian noise (AWGN) with specifed sigma
    print "Generating noisy image..."
    print "Noise standard deviation: %1.5f" % noise_sigma

    normal_noise = np.random.normal(scale=noise_sigma, size=orig_img.size)
    normal_noise = normal_noise.reshape(orig_img.shape)

    nois_img = orig_img + normal_noise

    misc.imsave("noisy.png", nois_img)

    # Normalize image, that is, translate values in image so its distribution
    # is comparable to a normal N(0, 1) (mean = 0.0, standard deviation = 1.0).
    # This way, parameters of the denoising algorithm, like h and sigma, are
    # independent of the values and distribution of the image.
    print "Normalizing noisy image..."
    nois_img_mean = nois_img.mean()
    nois_img_std  = nois_img.std()

    normal_nois_img = np.empty(nois_img.shape, dtype=np.float32)

    for x in xrange(normal_nois_img.shape[0]):
        for y in xrange(normal_nois_img.shape[1]):
            normal_nois_val = nois_img[x, y] - nois_img_mean
            if nois_img_std != 0.000001: normal_nois_val /= nois_img_std
            normal_nois_img[x, y] = normal_nois_val

    print "Denoising image..."
    normal_rest_img = denoise2D(normal_nois_img, True)

    print "Denormalizing noisy image..."
    rest_img = np.empty(nois_img.shape, dtype=orig_img.dtype)

    for x in xrange(rest_img.shape[0]):
        for y in xrange(rest_img.shape[1]):
            rest_val = normal_rest_img[x, y] * nois_img_std
            rest_val += nois_img_mean
            rest_img[x, y] = rest_val

    print "Storing denoised image..."
    misc.imsave("denoised.png", rest_img)

if __name__ == "__main__":
    main()
