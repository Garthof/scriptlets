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

[yy xx zz] = ndgrid(linspace(1, h, m(1)), ...
                    linspace(1, w, m(2)), ...
                    linspace(1, d, m(3)));

out_volume = interpn(in_volume, yy, xx, zz);

end
