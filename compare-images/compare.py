def mse(img1, img2):
    """
    Computes the mean squared error (MSE) of two images stored as
    numpy arrays. The arrays must contain float values between 0 and 1.
    """
    return ((img1 - img2) ** 2).mean(axis=None)


def psnr(img1, img2):
    """
    Computes the peak signal-to-noise ratio (PSNR) of two images stored as
    numpy arrays. The arrays must contain float values between 0 and 1.
    """
    import math
    return 10 * math.log(1. / mse(img1, img2), 10)
