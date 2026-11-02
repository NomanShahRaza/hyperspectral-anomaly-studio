function run_lrasr_detector(input_path, output_path)
%RUN_LRASR_DETECTOR Original axiqia main.m scoring path (dictionary + lrra).
% main.m builds a K-means / lowest-RX background dictionary, then solves
%   min ||Z||_* + lambda ||E||_{2,1}  s.t.  X = D Z + E
% with the original lrra.m LRR solver (LADMAP.m is the paper LRASR
% solver but is commented out in main.m). Score is the L2 column norm of E.
root = fileparts(mfilename('fullpath'));
addpath(root);
try
    pkg load statistics;
catch
end
data = load(input_path);
hsi = double(data.hsi_img);
hsi = hyperNormalize(hsi);
[h, w, bands] = size(hsi);
data1 = reshape(hsi, h * w, bands);
X = data1';

classum = 15;
P = 25;
lambda = 0.1;
if isfield(data, 'n_clusters') && ~isempty(data.n_clusters)
    classum = max(2, round(double(data.n_clusters)));
end
if isfield(data, 'atoms_per_cluster') && ~isempty(data.atoms_per_cluster)
    P = max(1, round(double(data.atoms_per_cluster)));
end
if isfield(data, 'lambda') && ~isempty(data.lambda)
    lambda = double(data.lambda);
end

try
    opts = statset('Display', 'off', 'MaxIter', 200);
    idx = kmeans(data1, classum, 'Options', opts);
catch
    idx = kmeans(data1, classum);
end
idx = idx(:);

D = [];
for k = 1:classum
    loc = find(idx == k);
    Nk = length(loc);
    if Nk < P
        continue;
    end
    kdata = data1(loc, :);
    Xk = kdata - repmat(mean(kdata), size(kdata, 1), 1);
    covk = (Xk' * Xk) ./ (size(Xk, 1) - 1);
    if rcond(covk) < 1e-12
        invcovk = pinv(covk);
    else
        invcovk = inv(covk);
    end
    PD = zeros(Nk, 1);
    for i = 1:Nk
        PD(i) = kdata(i, :) * invcovk * kdata(i, :)';
    end
    [~, index] = sort(PD);
    D = [D; kdata(index(1:P), :)];
end
if isempty(D)
    error('LRASR dictionary is empty (every K-means cluster had fewer than P pixels).');
end

[~, E] = lrra(X, D', lambda);
out = sqrt(sum(E .^ 2, 1));
out = reshape(out, h, w);
save(output_path, 'out', '-v7');
end
