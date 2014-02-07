%  This is the implementation of the RF filter of [Gastal and Oliveira 2011],
%  modified to use an l2-norm when blurring over the adaptive manifolds.
%
%
%  This code has been adapted from the reference implementation of the
%  adaptive-manifold high-dimensional filter described in the paper:
%
%    Adaptive Manifolds for Real-Time High-Dimensional Filtering
%    Eduardo S. L. Gastal  and  Manuel M. Oliveira
%    ACM Transactions on Graphics. Volume 31 (2012), Number 4.
%    Proceedings of SIGGRAPH 2012, Article 33.
%

function F = RF_filter(splat, manifold, sigma_s, sigma_r)

splat = double(splat);
manifold = double(manifold);

[h w d nn] = size(manifold);

dMcdx = diff(manifold, 1, 2);
dMcdy = diff(manifold, 1, 1);
dMcdz = diff(manifold, 1, 3);

dMdx = zeros(h,w,d);
dMdy = zeros(h,w,d);
dMdz = zeros(h,w,d);

for n = 1:nn
    dMdx(:,2:end,:) = dMdx(:,2:end,:) + ( dMcdx(:,:,:,n) ).^2;
    dMdy(2:end,:,:) = dMdy(2:end,:,:) + ( dMcdy(:,:,:,n) ).^2;
    dMdz(:,:,2:end) = dMdz(:,:,2:end) + ( dMcdz(:,:,:,n) ).^2;
end

sigma_H = sigma_s;

dHdx = sqrt((sigma_H/sigma_s).^2 + (sigma_H/sigma_r).^2 * dMdx);
dVdy = sqrt((sigma_H/sigma_s).^2 + (sigma_H/sigma_r).^2 * dMdy);
dTdz = sqrt((sigma_H/sigma_s).^2 + (sigma_H/sigma_r).^2 * dMdz);

dVdy = permute(dVdy, [3 1 2]);
dTdz = permute(dTdz, [2 3 1]);

N = 1;
F = splat;

for i = 0:N - 1
    sigma_H_i = sigma_H * sqrt(3) * 2^(N - (i + 1)) / sqrt(4^N - 1);

    F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i);
    F = permute(F, [3 1 2 4]);

    F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i);
    F = permute(F, [3 1 2 4]);

    F = TransformedDomainRecursiveFilter_Horizontal(F, dTdz, sigma_H_i);
    F = permute(F, [3 1 2 4]);

end

F = cast(F, class(splat));

end


function F = TransformedDomainRecursiveFilter_Horizontal(I, D, sigma)

a = exp(-sqrt(2) / sigma);

F = I;
V = a.^D;

[h w d nn] = size(I);

for c = 2:w
    for n = 1:nn
        F(:,c,:,n) = F(:,c,:,n) + V(:,c,:) .* (F(:,c-1,:,n) - F(:,c,:,n));
    end
end

for c = w-1:-1:1
    for n = 1:nn
        F(:,c,:,n) = F(:,c,:,n) + V(:,c+1,:) .* (F(:,c+1,:,n) - F(:,c,:,n));
    end
end

end
