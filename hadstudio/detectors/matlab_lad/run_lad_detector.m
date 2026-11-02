function run_lad_detector(input_path, output_path)
%RUN_LAD_DETECTOR Call original Verdoja LAD .m files.
root = fileparts(mfilename('fullpath'));
addpath(root);
data = load(input_path);
X = double(data.hsi_img);
detector = char(data.detector);
if strcmp(detector, 'lc')
    out = lad_C(X);
elseif strcmp(detector, 'lcs')
    out = lad_C_S(X);
elseif strcmp(detector, 'lq')
    out = lad_Q(X);
elseif strcmp(detector, 'lqs')
    out = lad_Q_S(X);
elseif strcmp(detector, 'rsad')
    alpha = 0.01;
    max_iter = 20;
    N = 80;
    L = 40;
    if isfield(data, 'alpha') && ~isempty(data.alpha)
        alpha = double(data.alpha);
    end
    if isfield(data, 'max_iter') && ~isempty(data.max_iter)
        max_iter = double(data.max_iter);
    end
    if isfield(data, 'n_blocks') && ~isempty(data.n_blocks)
        N = double(data.n_blocks);
    end
    if isfield(data, 'n_sel') && ~isempty(data.n_sel)
        L = double(data.n_sel);
    end
    try
        rng(0);
    catch
    end
    out = rsad(X, alpha, max_iter, N, L);
    out = double(out);
else
    error('Unknown LAD detector: %s', detector);
end
save(output_path, 'out', '-v7');
end
