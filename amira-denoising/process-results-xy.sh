#!/bin/bash

# Get directory with original files and directory to store the processed files
orig_dir=$1
proc_dir=$2

for file_name in $(ls $1); do
    # Remove extension from the file name
    img_name=${file_name%%.*}

    # Remove borders from the image and add a caption
    convert $1/$file_name -crop 578x687+222+109 -background White -pointsize 40 label:$img_name -gravity Center -append $2/$file_name
done

# Move to the processed directory and generate montages with the files. Each
# montage contains the results after an iteration of denoising. The images'
# sizes are not modified, and a border of 5 pixels is added between images.
cd $2
montage iter-001* -geometry +5+5 all-iter-001.png
montage iter-002* -geometry +5+5 all-iter-002.png
montage iter-003* -geometry +5+5 all-iter-003.png