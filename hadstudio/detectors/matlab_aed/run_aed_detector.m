function run_aed_detector(input_path, output_path)
%RUN_AED_DETECTOR Original AED demo.m scoring path (Kang et al. / yousefan).
root = fileparts(mfilename('fullpath'));
addpath(root);
addpath(fullfile(root, 'functions'));
try
    pkg load image;
catch
end
try
    pkg load statistics;
catch
end
data = load(input_path);
data1 = double(data.hsi_img);
para = 25;
n_dim = 3;
attr_type = 'a';
sigma_s = 5;
sigma_r = 0.5;
r = 2;
lamda = 100;
if isfield(data, 'para') && ~isempty(data.para)
    para = double(data.para);
end
if isfield(data, 'n_dim') && ~isempty(data.n_dim)
    n_dim = double(data.n_dim);
end
if isfield(data, 'attr_type') && ~isempty(data.attr_type)
    attr_type = char(data.attr_type);
end
if isfield(data, 'sigma_s') && ~isempty(data.sigma_s)
    sigma_s = double(data.sigma_s);
end
if isfield(data, 'sigma_r') && ~isempty(data.sigma_r)
    sigma_r = double(data.sigma_r);
end
if isfield(data, 'r') && ~isempty(data.r)
    r = double(data.r);
end
if isfield(data, 'lamda') && ~isempty(data.lamda)
    lamda = double(data.lamda);
end
if n_dim < 1
    n_dim = 1;
end
if para < 1
    para = 1;
end
data2 = NormalizeData(data1);
f0 = PCA_img(data1, n_dim);
f0 = NormalizeData(f0);
g0 = f0;
d0 = morph_detect(f0, para, attr_type);
out = RRX_detect(g0, d0, data2, sigma_s, sigma_r, r, lamda);
out = double(out);
save(output_path, 'out', '-v7');
end
