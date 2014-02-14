#!/usr/bin/env bash

get_current_patch_size() {
    local file_name=$1
    local patch_size

    patch_size=${file_name%%-pca*}          # Remove longest suffix matching -pca*
    patch_size=${patch_size##iter-*patch-}  # Remove longest preffix matching iter-*patch-

    echo $patch_size
}


compute_correct_patch_size() {
    local old_patch_size=$1
    local new_patch_size=$(echo "sqrt($old_patch_size-1) * 2 + 1" | bc)
    echo $new_patch_size
}


main() {
    local dir_name=$1
    local file_name

    for file_name in $dir_name/*.mat; do
        file_name=$(basename $file_name)
        local old_patch_size=$(get_current_patch_size $file_name)
        local new_patch_size=$(compute_correct_patch_size $old_patch_size)

        local file_name_preffix=${file_name%-*-pca*}
        local file_name_suffix=${file_name#iter-*patch-*-}
        local new_file_name=${file_name_preffix}-${new_patch_size}-${file_name_suffix}

        mv $dir_name/$file_name $dir_name/$new_file_name
    done
}

main $@
