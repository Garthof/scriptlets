#!/usr/bin/octave -qf

% Check args
if (nargin < 2)
    usage('%s', [program_name() ' input_file' ' output_file']);
end

% Read args
input_file = argv(){1};
output_file = argv(){2};

% Load file
input_file_data = load(input_file);

for [val key] = input_file_data
    in_volume = val;
    input_name = key;
end

% Convert volume to scale [0 1]
[in_volume min_value max_value] = vol2double(in_volume);

% Compute non-local-means patch space using 7x7 color patches reduced to
% the specified number of dimensions
patch_radius = 2  %patch size is 2*patch_radius + 1;
num_pca_dims = 6
patch_space = compute_non_local_means_basis(in_volume, patch_radius, ...
                                            num_pca_dims);

% Filtering parameters
sigma_s = 8
sigma_r = 0.35
num_pca_iters = 2;

% Compute tree height using Eq. (12)
tree_height = 2 + compute_manifold_tree_height(sigma_s, sigma_r)

% tilde_g is the output of our filter with outliers suppressed.
[output_volume tilde_output_volume] = adaptive_manifold_filter( ...
        in_volume, sigma_s, sigma_r, ...
        tree_height, patch_space, num_pca_iters);

% Save file
output_volume = double2vol(output_volume, min_value, max_value);
save('-mat-binary', output_file, 'output_volume');
