%  This is the low-pass filter 'h' we use for generating the
%  adaptive manifolds.
%
%
%  This code is based of the reference implementation of the adaptive-manifold
%  high-dimensional filter described in the paper:
%
%    Adaptive Manifolds for Real-Time High-Dimensional Filtering
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 31 (2012), Number 4.
%    Proceedings of SIGGRAPH 2012, Article 33.
%

function out_patch_vol = h_filter(in_patch_vol, sigma)

out_patch_vol = in_patch_vol;                              % a x b x c x d
out_patch_vol = h_filter_horizontal(out_patch_vol, sigma);
out_patch_vol = permute(out_patch_vol, [3 1 2 4]);         % c x a x b x d
out_patch_vol = h_filter_horizontal(out_patch_vol, sigma);
out_patch_vol = permute(out_patch_vol, [3 1 2 4]);         % b x c x a x d
out_patch_vol = h_filter_horizontal(out_patch_vol, sigma);
out_patch_vol = permute(out_patch_vol, [3 1 2 4]);         % a x b x c x d

end


function out_patch_vol = h_filter_horizontal(in_patch_vol, sigma)

a = exp(-sqrt(2) / sigma);

out_patch_vol = in_patch_vol;
[h w d nn] = size(in_patch_vol);

for c = 2:w
    out_patch_vol(:,c,:,:) = ...
            out_patch_vol(:,c,:,:) ...
            + a * (out_patch_vol(:,c-1,:,:) - out_patch_vol(:,c,:,:));
end

for c = w-1:-1:1
    out_patch_vol(:,c,:,:) = ...
            out_patch_vol(:,c,:,:) ...
            + a * (out_patch_vol(:,c+1,:,:) - out_patch_vol(:,c,:,:));
end

end
