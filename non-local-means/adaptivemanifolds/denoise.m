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
    input_volume = val;
    input_name = key;
end

% Convert volume to scale [0 1]
[input_volume min_value max_value] = vol2double(input_volume);

% Compute non-local-means patch space using 7x7 color patches reduced to
% the specified number of dimensions
patch_radius = 2;  %patch size is 2*patch_radius + 1;
num_pca_dims = 6;
nlmeans_space = compute_non_local_means_basis(input_volume, patch_radius, ...
                                              num_pca_dims);


% Save file
output_volume = input_volume;
save('-mat-binary', output_file, 'output_volume');
