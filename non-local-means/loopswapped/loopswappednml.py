import math
import numpy as np

from scipy import ndimage
from scipy import misc

# Initialize constants
window_size = 9                # Search window size - must be an odd number
patch_size = 5                 # Neighborhood size - must be an odd number
h = 0.5
sigma = 1.0

# Generate a Gauss kernel
kernel_size = patch_size
gauss_kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)

for y in xrange(-(kernel_size/2), kernel_size/2 + 1):
    for x in xrange(-(kernel_size/2), kernel_size/2 + 1):
        gauss_val = x*x + y*y
        gauss_val = math.exp(-gauss_val / (2.0 * sigma * sigma))
        gauss_val = gauss_val / math.sqrt(2.0 * math.pi * sigma * sigma)

        gauss_kernel[x + kernel_size/2, y + kernel_size/2] = gauss_val

def is_within(pos, shape):
    return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1]


def get_f(nois_img, pos):
    # f = np.zeros(nois_img.shape, dtype=np.float32)

    # for dx in xrange(-(window_size/2), window_size/2 + 1):
    #     for dy in xrange(-(window_size/2), window_size/2 + 1):
    #         disp_pos = (pos[0] + dx, pos[1] + dy)

    #         if is_within(disp_pos, f.shape):
    #             f[disp_pos] = 1.0

    # return f
    return np.ones(nois_img.shape, dtype=np.float32)


def get_vn(nois_img, un):
    vn = np.zeros(nois_img.shape, dtype=np.float32)

    for x in xrange(nois_img.shape[0]):
        for y in xrange(nois_img.shape[1]):
            for dx in xrange(-kernel_size/2, kernel_size/2 + 1):
                for dy in xrange(-kernel_size/2, kernel_size/2 + 1):
                    img_disp_pos = (x + dx, y + dy)
                    ker_disp_pos = (dx + kernel_size/2, dy + kernel_size/2)

                    if is_within(img_disp_pos, nois_img.shape):
                        vn[x,y] += gauss_kernel[ker_disp_pos] * nois_img[img_disp_pos]

    return vn

def get_un(nois_img, disp):
    un = np.zeros(nois_img.shape, dtype=nois_img.dtype)

    for x in xrange(nois_img.shape[0]):
        for y in xrange(nois_img.shape[1]):
            curr_pos = x, y
            disp_pos = x+disp[0], y+disp[1]

            if is_within(disp_pos, nois_img.shape):
                diff = nois_img[curr_pos] - nois_img[disp_pos]
                un[curr_pos] = diff * diff

    return un


def denoise2D(nois_img, verbose=False):
    denois_img  = np.zeros(nois_img.shape, dtype=nois_img.dtype)
    sum_weights = np.zeros(nois_img.shape, dtype=np.float32)

    for dx in xrange(-(window_size/2), window_size/2 + 1):
        for dy in xrange(-(window_size/2), window_size/2 + 1):
            un = get_un(nois_img, (dx, dy))
            vn = get_vn(nois_img, un)

            f = get_f(nois_img, (dx, dy))

            for x in xrange(nois_img.shape[0]):
                for y in xrange(nois_img.shape[1]):
                    weight = f[x, y] * math.exp(-vn[x, y] / (2*h*h))
                    sum_weights[x, y] += weight
                    denois_img[x, y]  += weight * nois_img[x, y]

    return (denois_img / sum_weights).astype(nois_img.dtype)


def main():
    print "Loading Lena image..."
    orig_img = misc.lena()[160:160+10, 160:160+10]

    print "Image dtype: %s" % orig_img.dtype
    print "Image size: %6d" % orig_img.size
    print "Image shape: %3dx%3d" % (orig_img.shape[0], orig_img.shape[1])
    print "Max value %3d at pixel %6d" % (orig_img.max(), orig_img.argmax())
    print "Min value %3d at pixel %6d" % (orig_img.min(), orig_img.argmin())
    print "Variance: %1.5f" % orig_img.var()
    print "Standard deviation: %1.5f" % orig_img.std()

    misc.imsave("orig.png", orig_img)

    print "Generating noisy image..."
    noisy_img   = orig_img \
                + 0.4 * orig_img.std() * np.random.random(orig_img.shape)

    misc.imsave("noisy.png", noisy_img)

    print "Denoising image..."
    denoised_img = denoise2D(orig_img, True)

    print "Storing denoised image..."
    misc.imsave("denoised.png", denoised_img)

if __name__ == "__main__":
    main()
