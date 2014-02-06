%  This implements a simple non-local-means patchspace. Note that this code is
%  not optimized for memory use.
%
%
%  This code is part of the reference implementation of the adaptive-manifold
%  high-dimensional filter described in the paper:
%
%    Adaptive Manifolds for Real-Time High-Dimensional Filtering
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 31 (2012), Number 4.
%    Proceedings of SIGGRAPH 2012, Article 33.

function [proj_patches eig_vals patches] = compute_non_local_means_basis( ...
                                                    input_volume, ...
                                                    patch_radius, ...
                                                    num_pca_dims)

% Center data around the mean
input_volume = input_volume - mean(mean(mean(input_volume)));

% Store patches in a cell array. Each cell has the same size as the volume, and
% contains a component of each patch. I tried another solution, but takes too
% much memory
nneighbors = (2*patch_radius + 1)^3;
patches = cell(nneighbors, 1);

for i = 1:nneighbors
    patches{i} = zeros(size(input_volume), 'single');
end

n = 1;
for i = -patch_radius:patch_radius
    for j = -patch_radius:patch_radius
        for k = -patch_radius:patch_radius
            dist2  = i^2 + j^2 + k^2;
            weight = exp(-dist2 / 2 / (patch_radius / 2));
            shifted_input_volume = circshift(input_volume, [i j k]);
            patches{n} = shifted_input_volume * weight;
            n = n + 1;
        end
    end
end

% The total number of patches is too high for my current version of Octave.
% I am running into some memory errors, so I am just picking some random
% patches from the volume and compute the PCA with them
num_rand_patches = 10^3;
rand_patches = zeros([num_rand_patches nneighbors]);
rand_indices = unidrnd(prod(size(input_volume)), [num_rand_patches 1]);

for i = 1:nneighbors
    rand_patches(:, i) = patches{i}(rand_indices);
end

% Perform the PCA
rand_patches = bsxfun(@minus, rand_patches, mean(rand_patches));

[eig_vecs eig_vals] = eig(rand_patches'*rand_patches);
eig_vecs = flipdim(eig_vecs, 2);
eig_vals = flipdim(diag(eig_vals),1)';
eig_vals = eig_vals(1:num_pca_dims);

% Project patches into the most important components of the eigenvectors
proj_patches = zeros([prod(size(input_volume)) num_pca_dims]);

for i = 1:prod(size(input_volume))
    for j = 1:nneighbors
        patch(j) = patches{j}(i);
    end

    proj_patches(i, :) = patch * eig_vecs(:, 1:num_pca_dims);
end

proj_patches = reshape(proj_patches, [size(input_volume) num_pca_dims]);

end
