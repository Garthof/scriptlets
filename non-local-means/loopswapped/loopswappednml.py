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


def is_within(img, pos):
    return (0 <= pos[0] < img.shape[0] and 0 <= pos[1] < img.shape[1])


def get_val(img, pos):
    if is_within(img, pos):
        return img[pos]
    else:
        return 0


def set_val(img, pos, val):
    if is_within(img, pos):
        img[pos] = val


def get_f(nois_img, pos):
    f = np.zeros(nois_img.shape, dtype=np.float32)

    for dx in xrange(-(window_size/2), window_size/2 + 1):
        for dy in xrange(-(window_size/2), window_size/2 + 1):
            disp_pos = (pos[0] + dx, pos[1] + dy)
            set_val(f, disp_pos, 1.0)

    return f
    # return np.ones(nois_img.shape, dtype=np.float32)


def get_vn(nois_img, un):
    vn = np.zeros(nois_img.shape, dtype=np.float32)

    for x in xrange(nois_img.shape[0]):
        for y in xrange(nois_img.shape[1]):
            vn_val  = get_val(vn, (x,y))

            for dx in xrange(-kernel_size/2, kernel_size/2 + 1):
                for dy in xrange(-kernel_size/2, kernel_size/2 + 1):
                    img_disp_pos = (x + dx, y + dy)
                    ker_disp_pos = (dx + kernel_size/2, dy + kernel_size/2)

                    ker_val = get_val(gauss_kernel, ker_disp_pos)
                    img_val = get_val(nois_img, img_disp_pos)

                    vn_val += ker_val * img_val

            set_val(vn, (x,y), vn_val)

    return vn


def get_un(nois_img, disp):
    un = np.zeros(nois_img.shape, dtype=nois_img.dtype)

    for x in xrange(nois_img.shape[0]):
        for y in xrange(nois_img.shape[1]):
            curr_pos = x, y
            disp_pos = x+disp[0], y+disp[1]

            curr_val = get_val(nois_img, curr_pos)
            disp_val = get_val(nois_img, disp_pos)

            diff = curr_val - disp_val
            set_val(un, curr_pos, diff * diff)

    return un


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

    weight  = (dist*dist) / (2*h*h)
    weight  = math.exp(-weight)

    return weight


def denoise2D(nois_img, verbose=False):
    denois_img  = np.zeros(nois_img.shape, dtype=nois_img.dtype)
    sum_weights = np.zeros(nois_img.shape, dtype=np.float32)

    for dx in xrange(-(window_size/2), window_size/2 + 1):
        for dy in xrange(-(window_size/2), window_size/2 + 1):
            for x in xrange(nois_img.shape[0]):
                for y in xrange(nois_img.shape[1]):
                    if is_within(nois_img,(x+dx,y+dy)):
                        weight = weight2D(nois_img, (x+dx,y+dy), (x,y))
                        sum_weights[x, y] += weight
                        denois_img[x, y]  += weight * nois_img[x+dx, y+dy]

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
