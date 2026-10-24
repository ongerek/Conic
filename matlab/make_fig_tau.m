% make_fig_tau.m
%
% Fig. 4 (Section 7.4): where training places the ratio tau.
% (a) tau over 100 epochs in the four settings, five seeds each, against the
%     path that weight decay alone would follow from tau = 1.5;
% (b) tau after 20 epochs against the learning rate, with and without batch
%     normalization, against the weight-decay-only values.
% Reads tau_traj.csv and tau_runs.csv (written by src/tau_replay.py); writes
% tau_trajectories.pdf and tau_trajectories.png. MATLAB and Octave.
% Run by typing  make_fig_tau  in the folder that holds the two CSVs.

function make_fig_tau()

T = readcsv('tau_traj.csv');                   % run, bn, lr0, epoch, lr, tau, tau_wd_only
R = readcsv('tau_runs.csv');

fig = figure('Color', 'w', 'Position', [100 100 1100 420]);

% ---------------- (a) trajectories ----------------
subplot(1, 2, 1);  hold on;
cells = {'S_conic_bn_s',          [0.80 0.15 0.10], '-',  'BN, lr 0.1';
         'CTRL_conic_bn_lr001_s', [0.80 0.15 0.10], '--', 'BN, lr 0.01';
         'S_conic_nobn_s',        [0.10 0.30 0.80], '-',  'no BN, lr 0.01';
         'LR8_conic_nobn_s',      [0.10 0.30 0.80], '--', 'no BN, lr 0.08'};
h = [];  hl = {};                               % settings without runs are left out
for c = 1:4
    for s = 0:4
        m = strcmp(T.run, sprintf('%s%d', cells{c,1}, s));
        if ~any(m), continue; end
        hh = plot(str2double(T.epoch(m)), str2double(T.tau(m)), cells{c,3}, ...
                  'Color', cells{c,2}, 'LineWidth', 0.9);
        if numel(hl) < c || isempty(hl{c}), h(end+1) = hh; hl{c} = cells{c,4}; end
    end
end
hl = hl(~cellfun(@isempty, hl));
% weight decay alone, for the three learning rates (identical within a rate);
% the paths for lr 0.1 and 0.08 end at the same value and share one legend entry
wd = {'CTRL_conic_bn_lr001_s0', [0.25 0.25 0.25], 'decay only, lr 0.01';
      'LR8_conic_nobn_s0',      [0.62 0.62 0.62], 'decay only, lr 0.08 and 0.1';
      'S_conic_bn_s0',          [0.62 0.62 0.62], 'decay only, lr 0.08 and 0.1'};
for k = 1:3
    m = strcmp(T.run, wd{k,1});
    if ~any(m), continue; end
    hh = plot(str2double(T.epoch(m)), str2double(T.tau_wd_only(m)), ':', ...
              'Color', wd{k,2}, 'LineWidth', 1.4);
    if ~any(strcmp(hl, wd{k,3})), h(end+1) = hh; hl{end+1} = wd{k,3}; end
end
plot([0 100], [1 1], 'k-', 'LineWidth', 0.5);
text(99, 1.05, '\tau = 1 (paraboloid)', 'FontSize', 8, 'HorizontalAlignment', 'right');
hold off;  box on;  xlim([0 100]);  ylim([0 1.6]);
xlabel('epoch');  ylabel('\tau  (mean over classes)');
if ~isempty(h), legend(h, hl, 'Location', 'southeast'); end
title('(a) \tau during training, five seeds per setting');

% ---------------- (b) learning-rate sweep, 20 epochs ----------------
subplot(1, 2, 2);  hold on;
isconic = strcmp(R.head, 'conic') & strncmp(R.run, 'LR_conic_', 9);
done    = str2double(R.epochs_done) == 20 & str2double(R.test_acc_final) > 0.11;
sel = isconic & done;
lr  = str2double(R.lr);  tf = str2double(R.tau_final);  tw = str2double(R.tau_wd_only);
bn  = strcmp(R.bn, 'yes');
[~, o] = sort(lr);
ht = [];  hd = [];  lt = {};  ld = {};
for grp = 1:2
    if grp == 1, m = sel & bn;  col = [0.80 0.15 0.10];  lab = 'BN';
    else,        m = sel & ~bn; col = [0.10 0.30 0.80];  lab = 'no BN'; end
    oo = o(m(o));
    if isempty(oo), continue; end
    ht(end+1) = semilogx(lr(oo), tf(oo), 'o-', 'Color', col, 'MarkerFaceColor', col, 'LineWidth', 1.0);
    hd(end+1) = semilogx(lr(oo), tw(oo), 's:', 'Color', col, 'LineWidth', 0.9);
    lt{end+1} = [lab ', trained'];
    ld{end+1} = [lab ', decay only'];
end
set(gca, 'XScale', 'log');
plot([0.004 0.8], [1 1], 'k-', 'LineWidth', 0.5);
hold off;  box on;  xlim([0.004 0.8]);  ylim([0 1.6]);
xlabel('learning rate');  ylabel('\tau after 20 epochs');
if ~isempty(ht), legend([ht hd], [lt ld], 'Location', 'northeast'); end
title('(b) \tau against learning rate');

set(fig, 'PaperPositionMode', 'auto');
pos = get(fig, 'Position');                      % pixels; 100 px per inch on paper
set(fig, 'PaperUnits', 'inches', 'PaperSize', pos(3:4)/100, ...
         'PaperPosition', [0 0 pos(3:4)/100]);
try
    exportgraphics(fig, 'tau_trajectories.pdf', 'ContentType', 'vector');   % MATLAB R2020a+
catch
    print(fig, '-dpdf', 'tau_trajectories.pdf');                           % older MATLAB, Octave
end
print(fig, '-dpng', '-r300', 'tau_trajectories.png');
fprintf('wrote tau_trajectories.pdf and tau_trajectories.png\n');
end


function S = readcsv(f)
% Read a CSV with a header line into a struct of cell arrays of strings.
fid = fopen(f, 'r');
if fid < 0, error('cannot open %s', f); end
hdr = strtrim(fgetl(fid));
names = strsplit(hdr, ',');
C = textscan(fid, repmat('%s', 1, numel(names)), 'Delimiter', ',', 'ReturnOnError', false);
fclose(fid);
for k = 1:numel(names)
    S.(strtrim(names{k})) = C{k};
end
end
