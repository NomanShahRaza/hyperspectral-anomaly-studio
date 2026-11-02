function run_frfe_detector(input_path, output_path)
%RUN_FRFE_DETECTOR Execute original FrFE demo.m scoring (D-RX, DWT-RX, FrFE).
root = fileparts(mfilename('fullpath'));
addpath(root);
try
    pkg load signal;
    pkg load image;
catch
end
data = load(input_path);
DataTest = double(data.hsi_img);
DataTest = DataTest ./ max(DataTest(:));
detector = char(data.detector);
[rows, cols, bands] = size(DataTest);
X = reshape(DataTest, rows * cols, bands);
M = rows * cols;
if strcmp(detector, 'drx')
    lag = double(data.lag);
    X2 = abs(X(:, 1 + lag:end) - X(:, 1:end - lag));
    r = RX(X2');
elseif strcmp(detector, 'dwtrx')
    XW = [];
    for i = 1:M
        try
            [ca, cd] = dwt(X(i, :), 'db1', 'mode', 'sym');
        catch
            lo_d = [1 / sqrt(2), 1 / sqrt(2)];
            hi_d = [-1 / sqrt(2), 1 / sqrt(2)];
            [ca, cd] = dwt(X(i, :), lo_d, hi_d);
        end
        XW(i, :) = [ca];
    end
    r = RX(XW');
elseif strcmp(detector, 'frfe')
    if isfield(data, 'order') && ~isempty(data.order) && isnumeric(data.order) && isfinite(data.order)
        order = double(data.order);
    else
        [~, order] = FRFEorder(DataTest);
    end
    im1 = zeros(rows, cols, bands);
    for i = 1:rows
        for j = 1:cols
            im1(i, j, :) = Disfrft(squeeze(DataTest(i, j, :)), order);
        end
    end
    im1 = center_standard(abs(im1));
    XT = reshape(im1, rows * cols, bands);
    r = RX(XT');
else
    error('Unknown FrFE detector: %s', detector);
end
out = reshape(r, [rows, cols]);
save(output_path, 'out', '-v7');
end
