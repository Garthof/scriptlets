#!/usr/bin/env bash

# Get directory with original files and directory to store the
# processed files
orig_dir=$1
proc_dir=$2

# Set the rotations to perform in the images
rotation_xy=270
rotation_xz=270
rotation_yz=90

# Set the dimensions to crop the images
# crop_dims_xy=612x660+206+122
# crop_dims_xz=550x706+235+98
# crop_dims_yz=577x683+222+110
crop_dims_xy=550x660+164+167
crop_dims_xz=550x660+177+157
crop_dims_yz=550x660+148+183

# Removes extension and prefix from the file name.
remove_prefix_sufix() {
    local file_name=$1
    local img_name

    img_name=${file_name%%.*}   # Remove longest suffix matching .*
    img_name=${img_name##*/}    # Remove longest prefix matching */

    echo $img_name
}

# Returns the orientation of the slice in the image.
get_orientation() {
    local file_name=$1
    local img_name
    local orientation

    img_name=$(remove_prefix_sufix $file_name)
    orientation=${img_name##*-} # Remove longest prefix matching *-

    echo $orientation
}

# Returns degrees of rotation to perform in the image.
get_rotation() {
    local file_name=$1
    local orientation
    local rotation

    orientation=$(get_orientation $file_name)

    if   [[ "$orientation" == "xy" ]]; then
        rotation=$rotation_xy
    elif [[ "$orientation" == "xz" ]]; then
        rotation=$rotation_xz
    elif [[ "$orientation" == "yz" ]]; then
        rotation=$rotation_yz
    else
        echo "$orientation in $file_name is unknown"
        exit
    fi

    echo $rotation
}

# Returns the dimensions to crop the image.
get_crop_dims() {
    local file_name=$1
    local orientation
    local crop_dims

    orientation=$(get_orientation $file_name)

    if   [[ "$orientation" == "xy" ]]; then
        crop_dims=$crop_dims_xy
    elif [[ "$orientation" == "xz" ]]; then
        crop_dims=$crop_dims_xz
    elif [[ "$orientation" == "yz" ]]; then
        crop_dims=$crop_dims_yz
    else
        echo "$orientation in $file_name is unknown"
        exit
    fi

    echo $crop_dims
}

# Rotates the image according to its orientation
rotate_image() {
    local file_name=$1
    local rotation

    rotation=$(get_rotation $file_name)

    mogrify -define png:format=png32 -rotate $rotation +repage $file_name
}

# Removes background from the slice image
crop_image() {
    local file_name=$1
    local crop_dims

    crop_dims=$(get_crop_dims $file_name)

    # Crop image. ImageMagick automatically converts the image into 8-bit
    # PNG based on the content, which in turn changes the gamma value. I
    # keep a 32-bit PNG mode (see http://goo.gl/BVbxjS). Also, cropping
    # leaves an offset in the PNG metadata, which further methods should
    # consider. To easen things, I remove the offset with +repage
    mogrify -define png:format=png32 -crop $crop_dims +repage $file_name
}

# Flips the image horizontally
flop_image() {
    local file_name=$1

    mogrify -define png:format=png32 -flop +repage $file_name
}

# Adds a label on the up-left corner with the orientation.
label_image() {
    local file_name=$1
    local orientation

    orientation=$(get_orientation $file_name)

    mogrify -define png:format=png32 -gravity NorthWest -pointsize 100 -stroke none -fill white -annotate 0x0+40+5 "$orientation" $file_name
}

# Puts together the three views in differents orientations of the same
# image
generate_set() {
    local file_name=$1
    local img_name

    # Get image name and remove orientation
    img_name=$(remove_prefix_sufix $file_name)
    img_name=${img_name%-*}   # Remove shortest suffix matching -*

    # Join images in the three possible orientations into a single one.
    # Once again, to avoid changing the gamma value, the 32-bit PNG mode
    # must be enforced (see http://goo.gl/BVbxjS)
    montage $proc_dir/$img_name-yz* $proc_dir/$img_name-xz* $proc_dir/$img_name-xy* -geometry +5+5 PNG32:$proc_dir/$img_name-set.png
}

# Generates a name for each set and labels the image with it
label_set() {
    local file_name=$1
    local img_name

    # Get image name and remove orientation
    img_name=$(remove_prefix_sufix $file_name)
    img_name=${img_name%-*}   # Remove shortest suffix matching -*

    # For some reason mogrify didn't work here, so I used convert
    convert $proc_dir/$img_name-set.png -gravity Center -background White -pointsize 100 label:$img_name -append PNG32:$proc_dir/$img_name-set.png
}

# Puts together sets belonging to the same iteration
generate_supersets() {
    montage $proc_dir/iter-001*-set.png -geometry +10+10 PNG32:$proc_dir/all-iter-001.png
    montage $proc_dir/iter-002*-set.png -geometry +10+10 PNG32:$proc_dir/all-iter-002.png
    montage $proc_dir/iter-003*-set.png -geometry +10+10 PNG32:$proc_dir/all-iter-003.png
}

main() {
    # Remove any previous result from the processed directory and copy
    # original images
    rm -rf $proc_dir/*
    cp -f $orig_dir/iter-*xy.png $proc_dir
    cp -f $orig_dir/iter-*xz.png $proc_dir
    cp -f $orig_dir/iter-*yz.png $proc_dir

    # Process each of the XY images in the directory
    for file_name in $proc_dir/*xy.png; do
        rotate_image $file_name
        crop_image $file_name
        label_image $file_name
    done

    # Process each of the XZ images in the directory
    for file_name in $proc_dir/*xz.png; do
        rotate_image $file_name
        crop_image $file_name
        label_image $file_name
    done

    # Process each of the YZ images in the directory
    for file_name in $proc_dir/*yz.png; do
        rotate_image $file_name
        crop_image $file_name
        flop_image $file_name
        label_image $file_name
    done

    # Generate sets and supersets
    for file_name in $proc_dir/*xy.png; do
        generate_set $file_name
        label_set $file_name
    done

    generate_supersets
}

main
