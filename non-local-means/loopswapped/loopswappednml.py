import math
import numpy as np

from scipy import ndimage
from scipy import misc

# Initialize constants
noise_sigma = 10.0
window_size = 21               # Search window size - must be an odd number
patch_size = 5                 # Neighborhood size - must be an odd number
h = 0.25
sigma = 1.0

# Generate a Gauss kernel
kernel_size = patch_size
gauss_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

for y in xrange(-(kernel_size/2), kernel_size/2 + 1):
    for x in xrange(-(kernel_size/2), kernel_size/2 + 1):
        gauss_val = x*x + y*y
        gauss_val = math.exp(-gauss_val / (2.0 * sigma * sigma))
        gauss_val = gauss_val / (2.0 * math.pi * sigma * sigma)

        gauss_kernel[x + kernel_size/2, y + kernel_size/2] = gauss_val


def is_within(img, pos):
    return (0 <= pos[0] < img.shape[0] and 0 <= pos[1] < img.shape[1])


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


def get_img_dists(img1, img2):
    img_dists = np.zeros(img1.shape, dtype=np.float32)

    for x in xrange(img_dists.shape[0]):
        for y in xrange(img_dists.shape[1]):
            patch1 = get_patch(img1, (x, y))
            patch2 = get_patch(img2, (x, y))
            img_dists[x, y] = get_dist(patch1, patch2)

    return img_dists


def get_img_weights(img1, img2):
    img_dists = get_img_dists(img1, img2)
    img_weights = np.exp(-img_dists / (2*h*h))

    return img_weights


def get_disp_img(orig_img, disp):
    disp_img = np.zeros(orig_img.shape, dtype=orig_img.dtype)

    for x in xrange(orig_img.shape[0]):
        for y in xrange(orig_img.shape[1]):
            disp_pos = x+disp[0], y+disp[1]
            if is_within(orig_img, (disp_pos)):
                disp_img[x, y] = orig_img[disp_pos]

    return disp_img


def denoise2D(nois_img, verbose=False):
    denois_img  = np.zeros(nois_img.shape, dtype=nois_img.dtype)
    sum_weights = np.zeros(nois_img.shape, dtype=np.float32)

    for dx in xrange(-(window_size/2), window_size/2 + 1):
        for dy in xrange(-(window_size/2), window_size/2 + 1):
            disp_img = get_disp_img(nois_img, (dx, dy))
            img_weights = get_img_weights(nois_img, disp_img)
            for x in xrange(nois_img.shape[0]):
                for y in xrange(nois_img.shape[1]):
                    if is_within(nois_img,(x+dx,y+dy)):
                        sum_weights[x, y] += img_weights[x, y]
                        denois_img[x, y]  += img_weights[x, y] * nois_img[x+dx, y+dy]

    return (denois_img / sum_weights).astype(nois_img.dtype)





    return (denois_img / sum_weights).astype(nois_img.dtype)


def get_noisy_img(orig_img):
    """
        Tries to load a file containing additive white Gaussian noise. If it
        does not exist, a file with noise is generated. The noise is added to
        the image.
    """
    noise_file_name = "awgn_nois.npy"

    try:
        normal_noise = np.load(noise_file_name)
        print "Load noise with standard deviation: %1.5f" % normal_noise.std()

    except IOError:
        print "Generate noise with standard deviation: %1.5f" % noise_sigma
        normal_noise = np.random.normal(scale=noise_sigma, size=orig_img.size)
        np.save(noise_file_name, normal_noise)

    normal_noise = normal_noise.reshape(orig_img.shape)

    return (orig_img + normal_noise)


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
    nois_img = get_noisy_img(orig_img)
    misc.imsave("noisy.png", nois_img)

    # Normalize image, that is, translate values in image so its distribution
    # is comparable to a normal N(0, 1) (mean = 0.0, standard deviation = 1.0).
    # This way, parameters of the denoising algorithm, like h and sigma, are
    # independent of the values and distribution of the image.
    print "Normalizing noisy image..."
    nois_img_mean = nois_img.mean()
    nois_img_std  = nois_img.std()

    normal_nois_img = nois_img - nois_img_mean
    if nois_img_std != 0.000001: # Divide by std. dev. if it is not zero
        normal_nois_img /= nois_img_std

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
