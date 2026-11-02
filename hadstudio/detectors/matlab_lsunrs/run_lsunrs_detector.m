function run_lsunrs_detector(input_path, output_path)
%RUN_LSUNRS_DETECTOR Call original UNRS / CRD / LSAD / LSUNRSORAD / LSAD-CR-IDW .m files.
root = fileparts(mfilename('fullpath'));
addpath(root);
data = load(input_path);
hsi = double(data.hsi_img);
detector = char(data.detector);
win_out = double(data.win_out);
win_in = double(data.win_in);
lambda = double(data.lambda);
if strcmp(detector, 'lsunrsorad')
    out = fun_LSUNRS_ED_ORAD(hsi, win_out, win_in, lambda);
elseif strcmp(detector, 'lsad_cr_idw')
    out = fun_LSAD_CR_IDW(hsi, win_out, win_in, lambda);
elseif strcmp(detector, 'unrs')
    out = UNRS_ID_Detect(hsi, win_out, win_in, lambda);
elseif strcmp(detector, 'crd')
    out = CRD_Detect(hsi, win_out, win_in, lambda);
elseif strcmp(detector, 'lsad')
    win_size = win_out;
    if isfield(data, 'win_size') && ~isempty(data.win_size)
        win_size = double(data.win_size);
    end
    if mod(win_size, 2) == 0
        win_size = win_size + 1;
    end
    out = fun_LSAD(hsi, win_size);
else
    error('Unknown LSUNRS detector: %s', detector);
end
save(output_path, 'out', '-v7');
end
