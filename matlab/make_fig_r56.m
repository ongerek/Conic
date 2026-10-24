% make_fig_r56.m
%
% Fig. 3 (Section 6): planes of the training loss through the whole backbone of
% ResNet-56 with and without skip connections. Writes fig_r56_skip.pdf and
% fig_r56_noskip.pdf (and PNG previews).
%
% Inputs: the two CSVs written by loss_surface.py --mode param --scope backbone
% --split train --grid 41 --images 512 (scripts/analyze.sh, step slices):
%   R56_skip_param_backbone_train.csv   R56_noskip_param_backbone_train.csv
% Works in MATLAB and Octave. Run it by typing  make_fig_r56  in the folder that
% holds the two CSVs.

function make_fig_r56()

isoct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
oldtk = '';                                     % Octave: the published figure used gnuplot
if isoct && any(strcmp(available_graphics_toolkits(), 'gnuplot'))
    oldtk = graphics_toolkit();  graphics_toolkit('gnuplot');
end
set(0, 'defaultaxesfontsize', 9, 'defaulttextfontsize', 9);

files  = {'R56_skip_param_backbone_train.csv', 'R56_noskip_param_backbone_train.csv'};
titles = {'ResNet-56, skips (train loss)', 'ResNet-56, no skips (train loss)'};
outs   = {'fig_r56_skip', 'fig_r56_noskip'};

for k = 1:numel(files)
    if ~exist(files{k}, 'file')
        fprintf('%s not found, skipped\n', files{k});
        continue
    end
    D = dlmread(files{k}, ',', 1, 0);
    x = unique(D(:,1));  g = numel(x);
    Z = reshape(D(:,3), g, g);          % rows along direction 2, columns along direction 1
    f = figure('Position', [60 60 760 330], 'Color', 'w');

    subplot(1, 2, 1);
    s = max(1, round(g/40));
    mesh(x(1:s:end), x(1:s:end), Z(1:s:end, 1:s:end));
    colormap(gray);  view(-37, 28);  axis tight;
    xlabel('direction 1');  ylabel('direction 2');  zlabel('loss');
    title(titles{k});

    subplot(1, 2, 2);
    contour(x, x, Z, 24, 'LineWidth', 0.7);  hold on;
    axis([x(1) x(end) x(1) x(end)]);  axis square;
    plot(0, 0, 'k+', 'MarkerSize', 9, 'LineWidth', 1.4);  hold off;
    xlabel('direction 1');  ylabel('direction 2');
    title('contours; + marks the trained point');

    if isoct
        print(f, [outs{k} '.pdf'], '-dpdf', '-S760,330');
    else
        set(f, 'PaperUnits', 'points', 'PaperSize', [760 330], 'PaperPosition', [0 0 760 330]);
        print(f, [outs{k} '.pdf'], '-dpdf');
    end
    print(f, [outs{k} '.png'], '-dpng', '-r150');
    fprintf('wrote %s.pdf and %s.png\n', outs{k}, outs{k});
    close(f);
end
if ~isempty(oldtk), graphics_toolkit(oldtk); end
end
