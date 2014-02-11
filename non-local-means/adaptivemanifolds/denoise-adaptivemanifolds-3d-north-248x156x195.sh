#!/usr/bin/env bash

source denoise-adaptivemanifolds.sh

global_variables() {
    global_script_path="/vis/data/people/bzflamas/scriptlets/non-local-means/adaptivemanifolds"
    global_script_name="denoise.m"

    global_num_iters="10"
    global_patch_radiuses="2 3 5"
    global_pca_dims="3 5 7"
    global_data_stddevs="0.1 0.3 0.5 1.0"
    global_spatial_stddevs="0.5 1.0 5.0 9.0"
    global_num_pca_iters="2"

    global_volume_name="Ammonit-Eo_u-cropped-248x156x195.vol.mat"
    global_experiment_name="denoise-adaptivemanifolds-3d-north-248x156x195"
    global_input_path="/vis/data/people/bzflamas/AmmoniteDenoising/datasets/volumes/Ammonit-Nano-CT/matlab/north-248x156x195"
    global_output_path="/vis/data/people/bzflamas/AmmoniteDenoising/datasets/volumes/output/$global_experiment_name"
    global_temp_path="/tmp"

    global_simulation_flag="0"
    global_sleep_seconds="3"
}

main() {
    set -e

    global_variables
    iterative_denoise
}

main $*
