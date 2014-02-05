#!/usr/bin/env bash

source denoise-gkdtrees.sh

global_variables() {
    global_amira_path="/vis/data/people/bzflamas/amira/product/bin"
    global_amira_exec="zibamira"
    global_amira_opt="-no_gui"

    global_script_path="/vis/data/people/bzflamas/scriptlets/amira-denoising"
    global_script_name="denoise-gkdtrees.hx"

    global_num_iters="10"
    global_patch_sizes="5 7 11"
    global_pca_dims="3 5 7"
    # global_data_stddevs="0.6 0.8 1.0"
    # global_spatial_stddevs="0.6 0.8 1.0"
    global_data_stddevs="1.0"
    global_spatial_stddevs="1.0"

    global_volume_name="Ammonit-Eo_u-cropped-141x152x180.vol.am"
    global_experiment_name="denoise-gkdtrees-3d-center-141x152x180"
    global_input_path="/vis/data/people/bzflamas/AmmoniteDenoising/datasets/volumes/Ammonit-Nano-CT/amira/center-141x152x180"
    global_output_path="/vis/data/people/bzflamas/AmmoniteDenoising/datasets/volumes/output/$global_experiment_name"
    global_temp_path="/tmp"

    global_simulation_flag="0"
    global_sleep_seconds="3"
    global_use_qsub="no"
}

main() {
    set -e

    global_variables
    iterative_denoise
}

main $*
