import numpy
import sys
import os

import tmpformat

def main():
    if len(sys.argv) != 2:
        print "Usage: convert_to_png <tmp_name>"
        exit()

    tmp_name = sys.argv[1]
    img_name, img_ext = os.path.splitext(tmp_name)

    imgs = tmpformat.load_from_tmp(tmp_name)

    for i in xrange(len(imgs)):
        if (len(imgs) == 1):
            png_name = "%s.%s" % (img_name, "png")
        else:
            png_name = "%s.%04i.%s" % (img_name, i, "png")

        tmpformat.misc.imsave(png_name, imgs[i])


if __name__ == "__main__":
    main()
