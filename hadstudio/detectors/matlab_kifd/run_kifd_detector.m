function run_kifd_detector(input_path, output_path)
%RUN_KIFD_DETECTOR Original KIFD demo.m scoring path (Li / Duan et al.).
root = fileparts(mfilename('fullpath'));
addpath(root);
addpath(fullfile(root, 'Kernel'));
try
    pkg load image;
catch
end
data = load(input_path);
data1 = double(data.hsi_img);
tree_num = 1000;
kpca_train = 10000;
kpca_dim = 300;
kernel_param = 1;
if isfield(data, 'tree_num') && ~isempty(data.tree_num)
    tree_num = double(data.tree_num);
end
if isfield(data, 'kpca_train') && ~isempty(data.kpca_train)
    kpca_train = double(data.kpca_train);
end
if isfield(data, 'kpca_dim') && ~isempty(data.kpca_dim)
    kpca_dim = double(data.kpca_dim);
end
if isfield(data, 'kernel_param') && ~isempty(data.kernel_param)
    kernel_param = double(data.kernel_param);
end
row = size(data1, 1);
col = size(data1, 2);
data2 = NormalizeData(data1);
data_kpca = kpca(data2, kpca_train, kpca_dim, 'Gaussian', kernel_param);
data = NormalizeData(data_kpca);
data = ToVector(data);
tree_size = floor(3 * row * col / 100);
if isfield(data, 'tree_size') && ~isempty(data.tree_size)
    tree_size = double(data.tree_size);
end
if tree_size < 2
    tree_size = 2;
end
s = iforest(data, tree_num, tree_size);
img = reshape(s, row, col);
stop_flag = 0;
index = [];
num = 1;
r0 = img;
lev = graythresh(r0);
while stop_flag == 0
    [r1, flag, s1, index1] = Local_iforest(r0, data, s, index, lev);
    r0 = r1;
    s = s1;
    index = index1;
    stop_flag = flag;
    num = num + 1;
    if num > 5
        break;
    end
end
index = (1:row * col)';
index(index1, :) = [];
Data_d = data(:, :);
Data_d(index1, :) = [];
if isempty(Data_d)
    r2 = 10 .^ r1;
else
    s_d = iforest(Data_d, tree_num, tree_size);
    r1(index) = s_d;
    r2 = 10 .^ r1;
end
out = r2;
save(output_path, 'out', '-v7');
rmpath(fullfile(root, 'Kernel'));
end
