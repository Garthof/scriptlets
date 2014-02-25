#!/usr/bin/env bash

global_variables() {
    global_script_path="/home/bzflamas/scriptlets/clustering"
    global_script_name="scicluster3d.py"

    global_max_iters="5 10 15"
    global_patch_radius="1 2"
    global_num_clusters="2 3 4 5 6 7 8"

    global_volume_name="equalized-contrast-188-uchar.mat"
    global_experiment_name="equalization-center-141x152x180"
    global_input_path="/media/data/bzflamas/AmmoniteDenoising/datasets/volumes/output/$global_experiment_name"
    global_output_path="/home/bzflamas/AmmoniteDenoising/datasets/volumes/output/$global_experiment_name"

    global_sleep_seconds="3"
}


iterative_clustering() {
    for max_iters in $global_max_iters; do
    for num_clusters in $global_num_clusters; do
    for patch_radius in $global_patch_radius; do
        local input_volume_path="$global_input_path"
        local input_volume_name="$global_volume_name"
        local input_volume="$input_volume_path/$input_volume_name"

        local output_volume_path="$global_output_path"
        local output_volume_name="$(get_volume_name $max_iters $num_clusters $patch_radius)"
        local output_volume="$output_volume_path/$output_volume_name"

        cluster_volume $input_volume $output_volume \
                       $max_iters $num_clusters $patch_radius
    done
    done
    done
}


cluster_volume() {
    local input_volume="$1"
    local output_volume="$2"
    local max_iters="$3"
    local num_clusters="$4"
    local patch_radius="$5"

    if [ ! -f $output_volume ]; then
        $global_script_path/$global_script_name \
                $input_volume $output_volume \
                -i $max_iters -n $num_clusters -p $patch_radius

        sleep $global_sleep_seconds
    else
        echo "Result $(basename $output_volume) already exists. Skipping..."
    fi
}


get_volume_name() {
    local max_iters="$1"
    local num_clusters="$2"
    local patch_radius="$3"

    printf -v formatted_iters_val "%03d" $max_iters
    printf -v formatted_clusters_val "%02d" $num_clusters
    printf -v formatted_patch_val "%02d" $(($patch_radius*2 + 1))

    local result=""
    result=$result"iters-${formatted_iters_val}"
    result=$result"-clusters-${formatted_clusters_val}"
    result=$result"-patch-${formatted_patch_val}"
    result=$result".mat"

    echo $result
}


main() {
    set -e
    global_variables $@
    iterative_clustering
}

main $@
