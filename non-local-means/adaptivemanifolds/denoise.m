#!/usr/bin/octave -qf

% Check args
if (nargin < 7)
    usage('%s', [program_name() ' input_file' ' output_file' 'patch_radius' ...
                 'num_pca_dims' 'sigma_s' 'sigma_r' 'num_pca_iters']);
end

% Read args
% Input and ouput files, in MATLAB/Octave format
input_file = argv(){1}
output_file = argv(){2}

% Patch size is 2*patch_radius + 1
patch_radius = str2double(argv(){3})

% Number of components kept from each patch
num_pca_dims = str2double(argv(){4})

% Spatial and range standard deviations
sigma_s = str2double(argv(){5})
sigma_r = str2double(argv(){6})

% Number of iters to quickly compute the eigenvector when building manifolds
num_pca_iters = str2double(argv(){7})


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
patch_space = compute_non_local_means_basis(in_volume, patch_radius, ...
                                            num_pca_dims);

% Compute tree height using Eq. (12)
tree_height = 2 + compute_manifold_tree_height(sigma_s, sigma_r)

% tilde_g is the output of our filter with outliers suppressed.
[output_volume tilde_output_volume] = adaptive_manifold_filter( ...
        in_volume, sigma_s, sigma_r, ...
        tree_height, patch_space, num_pca_iters);

% Save file
output_volume = double2vol(output_volume, min_value, max_value);
save('-mat-binary', output_file, 'output_volume');
