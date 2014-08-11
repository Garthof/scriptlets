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

    local orientation=$(get_orientation $file_name)

    if   [[ "$orientation" == "xy" ]]; then
        local crop_dims=$crop_dims_xy
    elif [[ "$orientation" == "xz" ]]; then
        local crop_dims=$crop_dims_xz
    elif [[ "$orientation" == "yz" ]]; then
        local crop_dims=$crop_dims_yz
    else
        echo "$orientation in $file_name is unknown"
        exit
    fi

    echo $crop_dims
}

# Rotates the image according to its orientation
rotate_image() {
    local file_name=$1

    local rotation=$(get_rotation $file_name)

    mogrify -define png:format=png32 -rotate $rotation +repage $file_name
}

# Removes background from the slice image
crop_image() {
    local file_name=$1

    local crop_dims=$(get_crop_dims $file_name)

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

    local orientation=$(get_orientation $file_name)

    mogrify -define png:format=png32 -gravity NorthWest -pointsize 30 -stroke none -fill white -annotate 0x0+10+5 "$orientation" $file_name
}

# Puts together the three views in differents orientations of the same
# image
generate_set() {
    local file_name=$1

    # Get image name and remove orientation
    local img_name
    img_name=$(remove_prefix_sufix $file_name)
    img_name=${img_name%-*}   # Remove shortest suffix matching -*

    # Join images in the three possible orientations into a single one.
    # Once again, to avoid changing the gamma value, the 32-bit PNG mode
    # must be enforced (see http://goo.gl/BVbxjS)
    local img_set
    [ "$direction" = "xy" ] && img_set="$proc_dir/$img_name-xy* $proc_dir/$img_name-xz* $proc_dir/$img_name-yz*"
    [ "$direction" = "xz" ] && img_set="$proc_dir/$img_name-xz* $proc_dir/$img_name-xy* $proc_dir/$img_name-yz*"
    [ "$direction" = "yz" ] && img_set="$proc_dir/$img_name-yz* $proc_dir/$img_name-xz* $proc_dir/$img_name-xy*"
    [ "$direction" = "3d" ] && img_set="$proc_dir/$img_name-xy* $proc_dir/$img_name-xz* $proc_dir/$img_name-yz*"

    montage $img_set -geometry +5+5 PNG32:$proc_dir/$img_name-set.png
}

# Generates a name for each set and labels the image with it
label_set() {
    local file_name=$1

    # Get image name and remove orientation
    local img_name
    img_name=$(remove_prefix_sufix $file_name)
    img_name=${img_name%-*}   # Remove shortest suffix matching -*

    # For some reason mogrify didn't work here, so I used convert
    convert $proc_dir/$img_name-set.png -gravity Center -background White -pointsize 25 label:$img_name -append PNG32:$proc_dir/$img_name-set.png
}

# Puts together sets belonging to the same iteration
generate_supersets() {
    montage $proc_dir/iter-001*-set.png -geometry +10+10 PNG32:$proc_dir/all-iter-001.png
    montage $proc_dir/iter-002*-set.png -geometry +10+10 PNG32:$proc_dir/all-iter-002.png
    montage $proc_dir/iter-003*-set.png -geometry +10+10 PNG32:$proc_dir/all-iter-003.png
}

# Processes each image to generate the final collage
process_results() {
    # Check orig_dir
    if [ -z "$orig_dir" ]; do
        echo "Please specify a non-empty path to retrieve original results"
        exit
    done

    # Check proc_dir before removing any content
    if [ -z "$proc_dir" ]; do
        echo "Please specify a non-empty path to store processed results"
        exit
    done

    if [ "$proc_dir" == "$HOME" ]; do
        echo "$HOME is not valid destination directory"
        exit
    done

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
        [ "$direction" = "xz" ] && flop_image $file_name
        label_image $file_name
    done

    # # Process each of the XZ images in the directory
    for file_name in $proc_dir/*xz.png; do
        rotate_image $file_name
        crop_image $file_name
        label_image $file_name
    done

    # Process each of the YZ images in the directory
    for file_name in $proc_dir/*yz.png; do
        rotate_image $file_name
        crop_image $file_name
        [ "$direction" = "yz" ] && flop_image $file_name
        label_image $file_name
    done

    # Generate sets and supersets
    for file_name in $proc_dir/*xy.png; do
        generate_set $file_name
        label_set $file_name
    done

    generate_supersets
}
