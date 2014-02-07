%  This code computes the manifold tree height for RGB color image filtering --
%  Eq. (10) of our paper.
%
%
%  This code has been adapted from the reference implementation of the
%  adaptive-manifold high-dimensional filter described in the paper:
%
%    Adaptive Manifolds for Real-Time High-Dimensional Filtering
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 31 (2012), Number 4.
%    Proceedings of SIGGRAPH 2012, Article 33.


function [Height K] = compute_manifold_tree_height(sigma_s, sigma_r)

Hs     = floor(log2(sigma_s)) - 1;
Lr     = 1 - sigma_r;

% Eq. (10)
Height = max(2, ceil(Hs .* Lr));
K      = 2.^Height - 1;

end
