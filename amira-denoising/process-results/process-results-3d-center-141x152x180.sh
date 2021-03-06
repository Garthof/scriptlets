#!/usr/bin/env bash

source process-results.sh

global_variables() {
    # Get directory with original files and directory to store the
    # processed files
    orig_dir=$1
    proc_dir=$2

    # Set direction
    direction="3d"

    # Set the rotations to perform in the images
    rotation_xy=0
    rotation_xz=0
    rotation_yz=0

    # Set the dimensions to crop the images
    crop_dims_xy=140x140+0+10
    crop_dims_xz=140x140+1+40
    crop_dims_yz=140x140+12+40
}

main() {
    global_variables $@
    process_results
}

main $@
