#!/usr/bin/env bash

# Get directory with original files and directory to store the processed files
orig_dir=$1
proc_dir=$2

# Process each of the images in the directory
for file_name in $(ls $1); do
    # Remove extension from the file name
    img_name=${file_name%%.*}

    # Remove borders from the image and add a caption. I tell IM to keep the
    # sRGB colorspace, although the resulting image is grey-valued. This
    # helps to keep the same gamma value as the original image. As seen in
    # http://goo.gl/znsL13
    convert $1/$file_name -crop 687x578+167+163 -rotate 270 -background White -pointsize 40 label:$img_name -gravity Center -append -colorspace sRGB $2/$file_name
done

# Move to the processed directory and generate montages with the files. Each
# montage contains the results after an iteration of denoising. The images'
# sizes are not modified, and a border of 5 pixels is added between images.
cd $2
montage iter-001* -geometry +5+5 -colorspace sRGB all-iter-001.png
montage iter-002* -geometry +5+5 -colorspace sRGB all-iter-002.png
montage iter-003* -geometry +5+5 -colorspace sRGB all-iter-003.png