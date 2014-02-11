iterative_denoise() {
    # Copy previous output to a temporary directory
    if [ -d $global_output_path ]; then
        cp -R $global_output_path $global_temp_path/$global_experiment_name.backup
    fi

    # Create output directory
    mkdir -p $global_output_path

    # Start iterative denoising
    for iter in $(seq 1 $global_num_iters); do
    for patch_radius in $global_patch_radiuses; do
    for pca_dim in $global_pca_dims; do
    for data_stddev in $global_data_stddevs; do
    for spatial_stddev in $global_spatial_stddevs; do
        # Set the correct input volume
        if [ $iter -eq 1 ]; then
            local input_volume_path="$global_input_path"
            local input_volume_name="$global_volume_name"
        else
            local input_volume_path="$global_output_path"
            local input_volume_name="$(get_volume_name $((iter-1)) $patch_radius $pca_dim $data_stddev $spatial_stddev).mat"
        fi

        local input_volume="$input_volume_path/$input_volume_name"

        # Set the output volume
        local output_volume_path="$global_output_path"
        local output_volume_name="$(get_volume_name $iter $patch_radius $pca_dim $data_stddev $spatial_stddev).mat"
        local output_volume="$output_volume_path/$output_volume_name"

        # Launch denoising algorithm
        denoise_volume  $input_volume $output_volume \
                        $patch_radius $pca_dim \
                        $data_stddev $spatial_stddev
    done
    done
    done
    done
    done
}


denoise_volume() {
    local input_volume="$1"
    local output_volume="$2"
    local patch_val="$3"
    local pca_val="$4"
    local data_stddev_val="$5"
    local spatial_stddev_val="$6"

    if [ ! -f $output_volume ]; then
        $global_script_path/$global_script_name \
                $input_volume $output_volume \
                $patch_val $pca_val \
                $spatial_stddev_val $data_stddev_val \
                $global_num_pca_iters

        sleep $global_sleep_seconds
    else
        echo "Result $(basename $output_volume) already exists. Skipping..."
    fi
}


get_volume_name() {
    local iter_val="$1"
    local patch_radius="$2"
    local pca_val="$3"
    local data_stddev_val="$4"
    local spatial_stddev_val="$5"

    printf -v formatted_iter_val "%03d" $iter_val
    printf -v formatted_patch_val "%02d" $(($patch_radius**2 + 1))
    printf -v formatted_pca_val "%03d" $pca_val
    printf -v formatted_data_stddev_val "%02.0f" $(echo "$data_stddev_val * 10.0" | bc)
    printf -v formatted_spatial_stddev_val "%02.0f" $(echo "$spatial_stddev_val * 10.0" | bc)

    echo "iter-${formatted_iter_val}-patch-${formatted_patch_val}-pca-${formatted_pca_val}-stddev-${formatted_data_stddev_val}-${formatted_spatial_stddev_val}"
}
