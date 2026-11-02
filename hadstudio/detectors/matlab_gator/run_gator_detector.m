function run_gator_detector(input_path, output_path)
%RUN_GATOR_DETECTOR Call original GatorSense CBAD, FCBAD, or CSD .m files.
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'anomaly_detectors'));
addpath(fullfile(root, 'util'));
data = load(input_path);
hsi_img = double(data.hsi_img);
detector = char(data.detector);
if isfield(data, 'n_clusters')
    n_cluster = double(data.n_clusters);
else
    n_cluster = 8;
end
if strcmp(detector, 'cbad')
    out = cbad_anomaly(hsi_img, [], n_cluster);
elseif strcmp(detector, 'fcbad')
    out = fcbad_anomaly(hsi_img, [], n_cluster);
elseif strcmp(detector, 'csd')
    n_dim_bg = double(data.n_dim_bg);
    if isfield(data, 'n_dim_tgt') && ~isempty(data.n_dim_tgt) && double(data.n_dim_tgt) > 0
        n_dim_tgt = double(data.n_dim_tgt);
    else
        n_dim_tgt = [];
    end
    tgt_orth = true;
    if isfield(data, 'tgt_orth')
        tgt_orth = logical(data.tgt_orth);
    end
    out = csd_anomaly(hsi_img, n_dim_bg, n_dim_tgt, tgt_orth);
else
    error('Unknown GatorSense detector: %s', detector);
end
save(output_path, 'out', '-v7');
end
