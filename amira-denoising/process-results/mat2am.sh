#!/usr/bin/env bash

global_variables() {
    global_dir_path=$1

    global_amira_path="/home/bzflamas/amira/product/bin"
    global_amira_exec="zibamira"
    global_amira_opt="-no_gui"

    global_script_path="/home/bzflamas/scriptlets/amira-denoising/process-results"
    global_script_name="mat2am.hx"
}

main() {
    global_variables $@

    for file_name in $global_dir_path/*.mat; do
        file_name_prefix=${file_name%.mat}

        export AMIRA_MATTOAM_INPUT_VOLUME="$file_name_prefix.mat"
        export AMIRA_MATTOAM_OUTPUT_VOLUME="$file_name_prefix.am"

        $global_amira_path/$global_amira_exec $global_amira_opt \
                $global_script_path/$global_script_name
    done
}

main $@
