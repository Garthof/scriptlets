#!/usr/bin/env bash

source process-results.sh

global_variables() {
    # Get directory with original files and directory to store the
    # processed files
    orig_dir=$1
    proc_dir=$2

    # Set direction
    direction="xz"

    # Set the rotations to perform in the images
    rotation_xy=180
    rotation_xz=0
    rotation_yz=270

    # Set the dimensions to crop the images
    crop_dims_xy=150x150+0+20
    crop_dims_xz=150x150+0+0
    crop_dims_yz=150x150+0+40
}

main() {
    global_variables $@
    process_results
}

main $@
