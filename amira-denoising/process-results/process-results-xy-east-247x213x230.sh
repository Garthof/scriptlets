#!/usr/bin/env bash

source process-results.sh

global_variables() {
    # Get directory with original files and directory to store the
    # processed files
    orig_dir=$1
    proc_dir=$2

    # Set direction
    direction="xy"

    # Set the rotations to perform in the images
    rotation_xy=0
    rotation_xz=0
    rotation_yz=0

    # Set the dimensions to crop the images
    crop_dims_xy=210x210+0+0
    crop_dims_xz=210x210+1+20
    crop_dims_yz=210x210+1+20
}

main() {
    global_variables $@
    process_results
}

main $@
