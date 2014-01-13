"""
Converts an image in the numpy.array format into the TMP format as defined
in [1], and stores the TMP file in disk. For more information on this quite
simple format, consult the source code available at http://goo.gl/WrpJct

The TMP format is derived from the internal format used by the ImageStack
library, which can be found in https://code.google.com/p/imagestack/.

[1]     Adams et al. "Gaussian KD-Trees for fast high-dimensional
        filtering" (2009)
"""

import array
import io
import numpy
import os.path
import struct
import sys

from scipy import ndimage
from scipy import misc


def load_from_tmp(file_name):
    """
    Returns a list of numpy arrays loaded from an image file in TMP format.
    Each array corresponds with a frame within the TMP image.
    """

    img_name = os.path.splitext(file_name)

    # Open and read the file
    with io.open(file_name, "rb") as bin_file:
        # Read header
        int_size = 4                # In TMP files each int is 4 bytes
        header = bin_file.read(4 * int_size)
        frames, width, height, channels = struct.unpack("iiii", header)

        # Read remaining data
        float_size = 4              # In TMP files each float is 4 bytes
        expected_data_size = frames * width * height * channels * float_size
        data = bin_file.read(expected_data_size)

        # Sanity check
        if len(data) != expected_data_size:
            raise Exception("File size does not match size in header")

    # Convert each frame into a numpy array
    imgs = []
    frame_size = width * height * channels * float_size

    for frame in xrange(0, frames):
        # Get data from the current frame
        cur_frame_pos = frame * frame_size
        nxt_frame_pos = cur_frame_pos + frame_size

        frame_data = data[cur_frame_pos:nxt_frame_pos]

        # Convert the sequence of bytes (a.k.a. string) into a one-dimensional
        # Python array of floats
        flat_arr = array.array('f')
        flat_arr.fromstring(frame_data)

        # Convert the Python array into a NumPy array with the proper
        # dimensions
        flat_img = numpy.array(flat_arr)

        if channels == 1:
            norm_img = flat_img.reshape((height, width))
        else:
            norm_img = flat_img.reshape((height, width, channels))

        # Convert image values to the range between 0 and 255
        img = norm_img * 255.0

        imgs.append(img.astype(numpy.uint8))

    return imgs


def save_to_tmp(file_name, imgs):
    """
    Saves numpy array stored in imgs into a file with TMP format.
    """
    frames = len(imgs)

    if frames == 0:
        raise Exception("The list of images is empty")

    # Get parameters for all images. The first image in the list is used
    # as reference
    img = imgs[0]
    num_dims = len(img.shape)

    if num_dims == 0:
        raise Exception("Images of 0 dimensions not supported!")

    if num_dims == 1:
        height   = 1
        width    = img.shape[0]
        channels = 1

    if num_dims == 2:
        height   = img.shape[0]
        width    = img.shape[1]
        channels = 1

    if num_dims == 3:
        height   = img.shape[0]
        width    = img.shape[1]
        channels = img.shape[2]

    if num_dims >= 4:
        print "Images of 4 or more dimensions not supported!"
        return

    # The header packs the number of frames, the width, the height, and the
    # number of channels as four integers
    header = struct.pack("iiii", frames, width, height, channels)

    with io.open(file_name, "wb") as bin_file:
        bin_file.write(header)

        for img in imgs:
            # Normalize image. Values should stay between 0 and 1
            norm_img = img / 255.0

            # Convert image into a Python array of floats. The image must be
            # reshaped first to have only one dimension
            flat_img = norm_img.flatten()
            flat_arr = array.array('f', flat_img)
            data = flat_arr.tostring()  # Get binary format (not pretty printed)

            # Write frame data into file
            bin_file.write(data)


def get_noise(noise_file_name="awgn_nois.npy", noise_size=None, noise_sigma=10.0):
    """
    Tries to load a file containing additive white Gaussian noise. If it
    does not exist, a file with noise is generated.
    """
    try:
        normal_noise = numpy.load(noise_file_name)
        print "Load noise with standard deviation: %1.5f" % normal_noise.std()

    except IOError:
        print "Generate noise with standard deviation: %1.5f" % noise_sigma
        normal_noise = numpy.random.normal(scale=noise_sigma, size=noise_size)
        numpy.save(noise_file_name, normal_noise)

    return normal_noise


def get_noisy_img(orig_img, noise):
    """
    Adds noise to the image.
    """
    # Reshape noise to the shape of the image
    noise =noise.reshape(orig_img.shape)

    # Add noise to the image
    noisy_img = orig_img + noise

    # Image values are expected to be between 0. and 255., remove values
    # outside that range. See http://goo.gl/UYDJLU
    noisy_img_flags = numpy.multiply(noisy_img >= 0., noisy_img <= 255.)
    noisy_img = numpy.multiply(noisy_img, noisy_img_flags) + 255. * (noisy_img > 255.)

    return noisy_img


def main():
    print "Loading Lena image..."
    # img_size = 50, 50
    # orig_img = misc.lena()[160:160+img_size[0], 160:160+img_size[1]]

    orig_img = misc.lena()

    print "Image dtype: %s" % orig_img.dtype
    print "Image size: %6d" % orig_img.size
    print "Image shape: %3dx%3d" % (orig_img.shape[0], orig_img.shape[1])
    print "Max value %3d at pixel %6d" % (orig_img.max(), orig_img.argmax())
    print "Min value %3d at pixel %6d" % (orig_img.min(), orig_img.argmin())
    print "Variance: %1.5f" % orig_img.var()
    print "Standard deviation: %1.5f" % orig_img.std()

    misc.imsave("orig.png", orig_img)

    # Generate additive white Gaussian noise (AWGN)
    print "Generating noisy image..."

    if len(sys.argv) == 2:
        # Use sigma specified by the user
        user_sigma = float(sys.argv[1])
        noise = get_noise(noise_size=orig_img.size, noise_sigma=user_sigma)
    else:
        # Use default sigma
        noise = get_noise(noise_size=orig_img.size)

    nois_img = get_noisy_img(orig_img, noise)
    misc.imsave("noisy.png", nois_img)

    # Normalize image, that is, translate values in image so its
    # distribution is comparable to a normal N(0, 1) (mean = 0.0,
    # standard deviation = 1.0). This way, parameters of the denoising
    # algorithm, like h and sigma, are independent of the values and
    # distribution of the image.
    print "Normalizing noisy image..."
    nois_img_mean = nois_img.mean()
    nois_img_std  = nois_img.std()

    normal_nois_img = nois_img - nois_img_mean
    if nois_img_std > 0.01: # Divide by std. dev. if it is not zero
        normal_nois_img /= nois_img_std

    # Test saving and loading a noisy image.
    print "Saving as TMP format..."
    save_to_tmp("noisy.tmp", normal_nois_img)

    print "Load from TMP format..."
    loaded_img = load_from_tmp("noisy.tmp")[0]

    # Check if saved and loaded image are quite similar (some small error
    # could be expected)
    if numpy.allclose(normal_nois_img, loaded_img):
        print "Saved and loaded images are equal"
    else:
        print "Saved and loaded images are NOT equal"


if __name__ == "__main__":
    main()
