iterative_denoise() {
    # Copy previous output to a temporary directory
    if [ -d $global_output_path ]; then
        cp -R $global_output_path $global_temp_path/$global_experiment_name.backup
    fi

    # Create output directory
    mkdir -p $global_output_path

    # Start iterative denoising
    for iter in $(seq 1 $global_num_iters); do
        for patch_size in $global_patch_sizes; do
            for window_size in $global_window_sizes; do
                for similarity_value in $global_similarity_values; do
                    # Set the correct input volume
                    if [ $iter -eq 1 ]; then
                        input_volume="$global_input_path/$global_volume_name"
                    else
                        input_volume="$global_output_path/$(get_volume_name $((iter-1)) $patch_size $window_size $similarity_value).am"
                    fi

                    # Set the output volume
                    output_volume_name="$(get_volume_name $iter $patch_size $window_size $similarity_value).am"
                    output_volume="$global_output_path/$output_volume_name"

                    # Launch denoising algorithm
                    denoise_volume $input_volume $output_volume $patch_size $window_size $similarity_value
                done
            done
        done
    done
}


denoise_volume() {
    input_volume="$1"
    output_volume="$2"
    patch_val="$3"
    window_val="$4"
    similarity_val="$5"

    if [ ! -e $output_volume ]; then
        if [ $global_use_qsub == "yes" ]; then
            # Generate a script file to launch the task
            script_file=$output_volume.pbs
            > $script_file

            echo "#PBS -o $output_volume.log" >> $script_file
            echo "#PBS -j yes" >> $script_file

            echo "export AMIRA_DENOISE_NLM_INPUT_VOLUME=$input_volume" >> $script_file
            echo "export AMIRA_DENOISE_NLM_OUTPUT_VOLUME=$output_volume" >> $script_file
            echo "export AMIRA_DENOISE_NLM_PATCH_SIZE=$patch_val" >> $script_file
            echo "export AMIRA_DENOISE_NLM_WINDOW_VALUE=$window_val" >> $script_file
            echo "export AMIRA_DENOISE_NLM_SIMILARITY_VALUE=$similarity_val" >> $script_file
            echo "export AMIRA_DENOISE_NLM_MODE=$global_mode_value" >> $script_file
            echo "export AMIRA_DENOISE_NLM_SIMULATION=$global_simulation_flag" >> $script_file

            echo "$global_amira_path/$global_amira_exec $global_amira_opt $global_script_path/$global_script_name" >> $script_file

            # Launch the task in the queue system
            qsub $script_file

        else
            export AMIRA_DENOISE_NLM_INPUT_VOLUME="$input_volume"
            export AMIRA_DENOISE_NLM_OUTPUT_VOLUME="$output_volume"
            export AMIRA_DENOISE_NLM_PATCH_SIZE="$patch_val"
            export AMIRA_DENOISE_NLM_WINDOW_VALUE="$window_val"
            export AMIRA_DENOISE_NLM_SIMILARITY_VALUE="$similarity_val"
            export AMIRA_DENOISE_NLM_MODE="$global_mode_value"
            export AMIRA_DENOISE_NLM_SIMULATION="$global_simulation_flag"

            set +e
            $global_amira_path/$global_amira_exec $global_amira_opt $global_script_path/$global_script_name
            set -e

            sleep $global_sleep_seconds
        fi
    else
        echo "Result $(basename $output_volume) already exists. Skipping..."
    fi
}


get_volume_name() {
    iter_val="$1"
    patch_val="$2"
    window_val="$3"
    similarity_val="$4"

    printf -v formatted_iter_val "%03d" $iter_val
    printf -v formatted_patch_val "%02d" $patch_val
    printf -v formatted_window_val "%04d" $window_val
    printf -v formatted_similarity_val "%02.0f" $(echo "$similarity_val * 10.0" | bc)

    echo "iter-$formatted_iter_val-patch-$formatted_patch_val-win-$formatted_window_val-simil-$formatted_similarity_val"
}
