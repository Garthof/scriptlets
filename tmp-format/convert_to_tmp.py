import sys
import tmpformat

def main():
    # Check arguments
    if len(sys.argv) <= 2:
        print "Usage: convert_to_tmp <tmp_name> <image_files>"
        exit()

    # Get arguments
    tmp_name = sys.argv[1]
    img_names = sys.argv[2:]

    # Load images
    imgs = []
    for img_name in img_names:
        imgs.append(tmpformat.misc.imread(img_name))

    # Convert image files into TMP format
    tmpformat.save_to_tmp(tmp_name, imgs)

if __name__ == "__main__":
    main()
