function AP = attribute_profile(I, A, L, C, F)
%ATTRIBUTE_PROFILE Area-profile fallback when the Dalla Mura MEX is absent.
% MATLAB loads attribute_profile.mexw64 instead. Octave uses this file.
% Matches the MEX layout used by EMAP_xdk: thickening, original, thinning.
if nargin < 4 || isempty(C)
    C = 4;
end
if nargin < 5 || isempty(F)
    F = 'min';
end
I = double(I);
L = double(sort(nonzeros(L(:)))');
if isempty(L)
    AP = I;
    return
end
nL = numel(L);
thin = zeros([size(I), nL]);
thick = zeros([size(I), nL]);
for k = 1:nL
    thin(:, :, k) = area_open(I, L(k), C);
    thick(:, :, k) = area_close(I, L(k), C);
end
AP = cat(3, thick(:, :, nL:-1:1), I, thin);
end

function J = area_open(I, lambda, conn)
hi = max(I(:));
if hi <= 0
    J = I;
    return
end
J = zeros(size(I));
if isequal(conn, 4)
    cc = 4;
else
    cc = 8;
end
lam = max(1, ceil(lambda));
for t = 1:ceil(hi)
    J = J + bwareaopen(I >= t, lam, cc);
end
end

function J = area_close(I, lambda, conn)
mx = max(I(:));
J = mx - area_open(mx - I, lambda, conn);
end
