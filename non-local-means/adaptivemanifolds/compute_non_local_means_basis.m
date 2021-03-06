%  This implements a simple non-local-means patchspace. Note that this code is
%  not optimized for memory use.
%
%
%  This code has been adapted from the reference implementation of the
%  adaptive-manifold high-dimensional filter described in the paper:
%
%    Adaptive Manifolds for Real-Time High-Dimensional Filtering
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 31 (2012), Number 4.
%    Proceedings of SIGGRAPH 2012, Article 33.

function [proj_patches eig_vals patches] = compute_non_local_means_basis( ...
                                                    in_volume, ...
                                                    patch_radius, ...
                                                    num_pca_dims)

% Initializate the random seed to the voxel in the center of the volume,
% as the original implementation did
rand('seed', in_volume(round(end/2),round(end/2),round(end/2)));

% Center data around the mean
in_volume = in_volume - mean(mean(mean(in_volume)));

% Store patches in a cell array. Each cell has the same size as the volume, and
% contains a component of each patch. I tried another solution, but takes too
% much memory
nneighbors = (2*patch_radius + 1)^3;
patches = cell(nneighbors, 1);

for i = 1:nneighbors
    patches{i} = zeros(size(in_volume), 'single');
end

n = 1;
for i = -patch_radius:patch_radius
    for j = -patch_radius:patch_radius
        for k = -patch_radius:patch_radius
            dist2  = i^2 + j^2 + k^2;
            weight = exp(-dist2 / 2 / (patch_radius / 2));
            shifted_in_volume = circshift(in_volume, [i j k]);
            patches{n} = shifted_in_volume * weight;
            n = n + 1;
        end
    end
end

% The total number of patches is too high for my current version of Octave.
% I am running into some memory errors, so I am just picking some random
% patches from the volume and compute the PCA with them.
num_rand_patches = 10^3;
rand_patches = zeros([num_rand_patches nneighbors]);
rand_indices = unidrnd(prod(size(in_volume)), [num_rand_patches 1]);

for i = 1:nneighbors
    rand_patches(:, i) = patches{i}(rand_indices);
end

% Perform the PCA
rand_patches = bsxfun(@minus, rand_patches, mean(rand_patches));

[eig_vecs eig_vals] = eig(rand_patches'*rand_patches);
eig_vecs = flipdim(eig_vecs, 2);
eig_vals = flipdim(diag(eig_vals),1)';
eig_vals = eig_vals(1:num_pca_dims);

% Project patches into the most significative eigenvectors. Each patch should
% be multiplied by the eigenvectors; however, as the patch contents are
% distributed in cell arrays, the multiplication must be performed in several
% iterations of a loop, adding the components each time
proj_patches = zeros([prod(size(in_volume)) num_pca_dims]);

for i = 1:nneighbors
    patch = reshape(patches{i}, [prod(size(in_volume)) 1]);
    patch = patch * eig_vecs(i, 1:num_pca_dims);
    proj_patches = proj_patches + patch;
end

proj_patches = reshape(proj_patches, [size(in_volume) num_pca_dims]);

end
