#!/usr/bin/env bash

source denoise-nlm.sh

global_variables() {
    global_amira_path="/home/bzflamas/amira/product/bin"
    global_amira_exec="zibamira"
    global_amira_opt="-no_gui"

    global_script_path="/home/bzflamas/scriptlets/amira-denoising"
    global_script_name="denoise-nlm.hx"

    global_num_iters="3"
    global_patch_sizes="5 7 11"
    global_window_sizes="11 21 31"
    global_similarity_values="0.8 0.9 1.0 1.1 1.2"
    global_mode_value="3D"

    global_volume_name="Ammonit-Eo_u-cropped-247x213x230.vol.am"
    global_experiment_name="denoise-nlm-3d-east-247x213x230"
    global_input_path="/home/bzflamas/AmmoniteDenoising/datasets/volumes/Ammonit-Nano-CT/amira/east-247x213x230"
    global_output_path="/home/bzflamas/AmmoniteDenoising/datasets/volumes/output/$global_experiment_name"
    global_temp_path="/tmp"

    global_simulation_flag="0"
    global_sleep_seconds="3"
    global_use_qsub="no"
}

main() {
    global_variables

    iterative_denoise
}

set -e
main $*
