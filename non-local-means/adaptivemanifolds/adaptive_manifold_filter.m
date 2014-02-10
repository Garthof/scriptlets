% High-dimensional filtering using adaptive manifolds
%
%  Parameters:
%    f                    Input volume to be filtered.
%    sigma_s              Filter spatial standard deviation.
%    sigma_r              Filter range standard deviation.
%
%  Optional parameters:
%    tree_height          Height of the manifold tree
%    patch_space          Space of all possible patches in the volume.
%    num_pca_iters        Number of iterations to compute the eigenvector v1.
%
%  Output:
%    out_volume           Adaptive-manifold filter response adjusted for
%                         outliers.
%    tilde_out_volume     Adaptive-manifold filter response NOT adjusted for
%                         outliers.
%
%
%  This code has been adapted from the reference implementation of the
%  adaptive-manifold high-dimensional filter described in the paper:
%
%    Adaptive Manifolds for Real-Time High-Dimensional Filtering
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 31 (2012), Number 4.
%    Proceedings of SIGGRAPH 2012, Article 33.

function [out_volume tilde_out_volume] = adaptive_manifold_filter( ...
                in_volume, sigma_s, sigma_r, ...
                tree_height, patch_space, ...
                num_pca_iters)

in_volume = vol2double(in_volume);

global sum_w_ki_Psi_blur;
global sum_w_ki_Psi_blur_0;

sum_w_ki_Psi_blur   = zeros(size(in_volume));
sum_w_ki_Psi_blur_0 = zeros(size(in_volume));

global min_pixel_dist_to_manifold_squared;

min_pixel_dist_to_manifold_squared = inf(size(in_volume));

global tree_nodes_visited;
tree_nodes_visited = 0;

% Algorithm 1, Step 1: compute the first manifold by low-pass filtering.
eta_1     = h_filter(patch_space, sigma_s);
cluster_1 = true(size(in_volume));

current_tree_level = 1;

build_manifolds_and_perform_filtering(...
       in_volume ...
     , patch_space ...
     , eta_1 ...
     , cluster_1 ...
     , sigma_s ...
     , sigma_r ...
     , current_tree_level ...
     , tree_height ...
     , num_pca_iters ...
 );

% Compute the filter response by normalized convolution -- Eq. (4)
tilde_out_volume = bsxfun(@rdivide, sum_w_ki_Psi_blur, sum_w_ki_Psi_blur_0);

% Adjust the filter response for outlier pixels -- Eq. (10)
alpha = exp( -0.5 .* min_pixel_dist_to_manifold_squared ./ sigma_r ./ sigma_r );
out_volume     = in_volume + bsxfun(@times, alpha, tilde_out_volume - in_volume);

% Close progressbar
delete(waitbar_handle);

end

function build_manifolds_and_perform_filtering(...
       in_volume ...
     , patch_space ...
     , eta_k ...
     , cluster_k ...
     , sigma_s ...
     , sigma_r ...
     , current_tree_level ...
     , tree_height ...
     , num_pca_iters ...
)

% Dividing the covariance matrix by 2 is equivalent to dividing
% the standard deviations by sqrt(2).
sigma_r_over_sqrt_2 = sigma_r / sqrt(2);

%% Compute downsampling factor
floor_to_power_of_two = @(r) 2^floor(log2(r));
df = min(sigma_s / 4, 256 * sigma_r);
df = floor_to_power_of_two(df);
df = max(1, df);

[h_vol w_vol d_vol] = size(in_volume);
rvol_size = round(size(in_volume) / df);

downsample = @(x) patch_space_scale( ...
                        x, [rvol_size(1) rvol_size(2) rvol_size(3)], 'linear');
upsample   = @(x) patch_space_scale( ...
                        x, [h_vol w_vol d_vol], 'linear');

%% Splatting: project the pixel values onto the current manifold eta_k

phi = @(x_squared, sigma) exp(-0.5 .* x_squared ./ sigma / sigma);

if size(eta_k,1) == size(patch_space,1)
    X = patch_space - eta_k;
    eta_k = downsample(eta_k);
else
    X = patch_space - upsample(eta_k);
end

% Project pixel colors onto the manifold -- Eq. (3), Eq. (5)
pixel_dist_to_manifold_squared = sum(X.^2, 4);
gaussian_distance_weights      = phi(pixel_dist_to_manifold_squared, ...
                                     sigma_r_over_sqrt_2);
Psi_splat                      = bsxfun(@times, gaussian_distance_weights, ...
                                        in_volume);
Psi_splat_0                    = gaussian_distance_weights;

% Save min distance to later perform adjustment of outliers -- Eq. (10)
global min_pixel_dist_to_manifold_squared;
min_pixel_dist_to_manifold_squared = ...
    min(min_pixel_dist_to_manifold_squared, pixel_dist_to_manifold_squared);

%% Blurring: perform filtering over the current manifold eta_k

blurred_projected_values = RF_filter(...
      downsample(cat(4, Psi_splat, Psi_splat_0)) ...
    , eta_k ...
    , sigma_s / df ...
    , sigma_r_over_sqrt_2 ...
);

w_ki_Psi_blur   = blurred_projected_values(:,:,:,1:end-1);
w_ki_Psi_blur_0 = blurred_projected_values(:,:,:,end);

%% Slicing: gather blurred values from the manifold

global sum_w_ki_Psi_blur;
global sum_w_ki_Psi_blur_0;

% Since we perform splatting and slicing at the same points over the manifolds,
% the interpolation weights are equal to the gaussian weights used for splatting.
w_ki = gaussian_distance_weights;

sum_w_ki_Psi_blur   = sum_w_ki_Psi_blur   ...
                    + bsxfun(@times, w_ki, upsample(w_ki_Psi_blur  ));
sum_w_ki_Psi_blur_0 = sum_w_ki_Psi_blur_0 ...
                    + bsxfun(@times, w_ki, upsample(w_ki_Psi_blur_0));

%% Compute two new manifolds eta_minus and eta_plus

%%% HERE I AM
global tree_nodes_visited;
tree_nodes_visited = tree_nodes_visited + 1;
% waitbar(tree_nodes_visited / (2^tree_height - 1), waitbar_handle);

% Test stopping criterion
if current_tree_level < tree_height

    % Algorithm 1, Step 2: compute the eigenvector v1
    X  = reshape(X, [h_vol*w_vol dR_joint]);
    rand_vec = rand(1,size(X,2)) - 0.5;
    v1 = compute_eigenvector(X(cluster_k(:),:), num_pca_iters, rand_vec);

    % Algorithm 1, Step 3: Segment pixels into two clusters -- Eq. (6)
    dot = reshape(X * v1', [h_vol w_vol]);
    cluster_minus = logical((dot <  0) & cluster_k);
    cluster_plus  = logical((dot >= 0) & cluster_k);

    % Algorithm 1, Step 4: Compute new manifolds by weighted low-pass filtering -- Eq. (7-8)
    theta = 1 - w_ki;

    eta_minus = bsxfun(@rdivide ...
        , h_filter(downsample(bsxfun(@times, cluster_minus .* theta, patch_space)), sigma_s / df) ...
        , h_filter(downsample(               cluster_minus .* theta          ), sigma_s / df));

    eta_plus = bsxfun(@rdivide ...
        , h_filter(downsample(bsxfun(@times, cluster_plus .* theta, patch_space)), sigma_s / df) ...
        , h_filter(downsample(               cluster_plus .* theta          ), sigma_s / df));

	% Only keep required data in memory before recursing
    keep in_volume patch_space eta_minus eta_plus cluster_minus cluster_plus sigma_s sigma_r current_tree_level tree_height num_pca_iters

    % Algorithm 1, Step 5: recursively build more manifolds.
    build_manifolds_and_perform_filtering(in_volume, patch_space, eta_minus, cluster_minus, sigma_s, sigma_r, current_tree_level + 1, tree_height, num_pca_iters);
    keep in_volume patch_space eta_plus cluster_plus sigma_s sigma_r current_tree_level tree_height num_pca_iters
    build_manifolds_and_perform_filtering(in_volume, patch_space, eta_plus,  cluster_plus,  sigma_s, sigma_r, current_tree_level + 1, tree_height, num_pca_iters);
end

end

% This function implements a O(dR N) algorithm to compute the eigenvector v1
% used for segmentation. See Appendix B.
function p = compute_eigenvector(X, num_pca_iters, rand_vec)

p = rand_vec;

for i = 1:num_pca_iters

    dots = sum( bsxfun(@times, p, X), 2 );
    t = bsxfun(@times, dots, X);
    t = sum(t, 1);
    p = t;

end

p = p / norm(p);

end


function ret = imresize_wrapper(im, m, interp)

[row, col, num_planes, tmp] = size(im);

% Resize plane by plane
for plane = 1:num_planes
    ret(:, :, plane) = imresize(im(:, :, plane), m, interp);
end

end


% Resize each component of the patch space separatedly
function out_patch_space = patch_space_scale(in_patch_space, m, op)

[h w d nn] = size(in_patch_space);

for n = 1:nn
    out_patch_space(:,:,:,n) = volscale(in_patch_space(:,:,:,n), m, op);
end

end


% Resize a volume of data using interp3. Kudos to Martin, who pointed me to
% a possible solution, here: http://stackoverflow.com/a/12521658/1679
function out_volume = volscale(in_volume, m, op)

[h w d] = size(in_volume);

[xx yy zz] = ndgrid(linspace(1, h, m(1)), ...
                    linspace(1, w, m(2)), ...
                    linspace(1, d, m(3)));

out_volume = interp3(in_volume, xx, yy, zz);

end
