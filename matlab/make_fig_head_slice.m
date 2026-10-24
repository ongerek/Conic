% make_fig_head_slice.m
%
% Fig. 2 (Section 5): the loss on a plane through the trained point along which
% only the affine final layer moves, ResNet-18 with batch normalization.
% Writes fig_head_slice.pdf and fig_head_slice.png.
%
% Inputs: the two CSVs written by loss_surface.py (--mode param --scope head) at
% half-widths 40 and 150 (scripts/analyze.sh, step slices). Works in MATLAB and
% Octave. Run it by typing  make_fig_head_slice  in the folder that holds the CSVs.

function make_fig_head_slice()

isoct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
oldtk = '';                                     % Octave: the published figure used gnuplot
if isoct && any(strcmp(available_graphics_toolkits(), 'gnuplot'))
    oldtk = graphics_toolkit();  graphics_toolkit('gnuplot');
end

f40  = 'S_affine_bn_s0_param_head_span40.csv';
f150 = 'S_affine_bn_s0_param_head_span150.csv';

[X40,  Y40,  Z40 ] = readslice(f40);
[X150, Y150, Z150] = readslice(f150);

fig = figure('Color', 'w', 'Position', [100 100 900 760]);

% (a) the wide surface
subplot(2, 2, 1);
h = mesh(X150, Y150, Z150);
set(h, 'EdgeColor', [0.25 0.25 0.25], 'FaceColor', 'w');
xlabel('direction 1');  ylabel('direction 2');  zlabel('loss');
title('(a) final layer only, span 150');
view(-37.5, 30);  axis tight;  grid on;

% (b) its contours: far out they are dilates of one convex curve
subplot(2, 2, 2);
contour(X150, Y150, Z150, linspace(min(Z150(:)), max(Z150(:)), 24), 'k');
hold on;  plot(0, 0, 'k+', 'MarkerSize', 9, 'LineWidth', 1.4);  hold off;
axis equal tight;  box on;
xlabel('direction 1');  ylabel('direction 2');
title('(b) contours, span 150');

% (c) the core, where the log-sum-exp rounds the apex
subplot(2, 2, 3);
contour(X40, Y40, Z40, linspace(min(Z40(:)), max(Z40(:)), 24), 'k');
hold on;  plot(0, 0, 'k+', 'MarkerSize', 9, 'LineWidth', 1.4);  hold off;
axis equal tight;  box on;
xlabel('direction 1');  ylabel('direction 2');
title('(c) contours, span 40');

% (d) chord slopes along exact grid rays: nondecreasing, saturating at the
%     slope of the asymptotic cone (Proposition 10)
subplot(2, 2, 4);
dirs = [1 0; 0 1; -1 0; 0 -1; 1 1; -1 1; -1 -1; 1 -1; ...
        2 1; 1 2; -2 1; -1 2; -2 -1; -1 -2; 2 -1; 1 -2];
c  = (size(Z150, 1) + 1) / 2;                 % index of the trained point
hx = X150(1, 2) - X150(1, 1);                 % grid step
L0 = Z150(c, c);
hold on;
worst = Inf;
for m = 1:size(dirs, 1)
    dx = dirs(m, 1);  dy = dirs(m, 2);
    n  = floor((c - 1) / max(abs(dx), abs(dy)));
    k  = 1:n;
    L  = Z150(sub2ind(size(Z150), c + dy*k, c + dx*k));
    r  = k * hypot(dx, dy) * hx;
    cs = (L - L0) ./ r;
    worst = min(worst, min(diff(cs)));
    plot(r, cs, 'k.-', 'MarkerSize', 6, 'LineWidth', 0.6);
end
hold off;  box on;  grid on;
xlabel('distance from the trained point');
ylabel('chord slope  (L(r) - L(0)) / r');
title('(d) chord slopes along 16 rays');

fprintf('smallest increment of a chord slope along a ray: %.3g (nondecreasing if >= 0)\n', worst);

set(fig, 'PaperPositionMode', 'auto');
try
    exportgraphics(fig, 'fig_head_slice.pdf', 'ContentType', 'vector');   % MATLAB R2020a+
catch
    print(fig, '-dpdf', 'fig_head_slice.pdf');                           % older MATLAB, Octave
end
print(fig, '-dpng', '-r200', 'fig_head_slice.png');
fprintf('wrote fig_head_slice.pdf and fig_head_slice.png\n');
close(fig);
if ~isempty(oldtk), graphics_toolkit(oldtk); end
end


function [X, Y, Z] = readslice(f)
% Read x,y,z from a loss_surface.py CSV and rebuild the full grid.
M  = dlmread(f, ',', 1, 0);
x  = M(:,1);  y = M(:,2);  z = M(:,3);
xs = unique(x);  ys = unique(y);
[X, Y] = meshgrid(xs, ys);
Z  = nan(size(X));
[~, ix] = ismember(x, xs);  [~, iy] = ismember(y, ys);
Z(sub2ind(size(Z), iy, ix)) = z;
if any(isnan(Z(:)))
    error('%s is not a full rectangular grid.', f);
end
end
