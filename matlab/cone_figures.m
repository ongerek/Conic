% cone_figures.m
%
% Fig. 1 of the manuscript (Section 4). Runs in MATLAB and in GNU Octave.
%
%   cone_envelope.pdf  the score surface min_k (g_k - rho_k) over a two-dimensional
%                      feature space. This surface is the lower envelope of the cones.
%   cone_cells.pdf     the same object seen from above: the cells and the creases
%                      where the envelope switches from one cone to the next. The
%                      straight line through the ends of the most curved crease is
%                      drawn dashed for comparison.
%
% No data are needed: the four cones are fixed below. Run by typing  cone_figures
% in the folder where the two PDFs should be written.

clear; close all;
isoct = exist('OCTAVE_VERSION', 'builtin') ~= 0;
oldtk = '';                                     % Octave: the published figure used gnuplot
if isoct && any(strcmp(available_graphics_toolkits(), 'gnuplot'))
    oldtk = graphics_toolkit();  graphics_toolkit('gnuplot');
end
% font sizes 15/14 give the published look in Octave; in MATLAB, 11/10 looks similar
if isoct
    set(0, 'defaulttextfontsize', 15, 'defaultaxesfontsize', 14);
else
    set(0, 'defaulttextfontsize', 11, 'defaultaxesfontsize', 10);
end

% ---- four cone functions  g_k(z) - rho_k = <w_k, z-a_k> + alpha_k*|z-a_k| - rho_k ----
% unequal apertures and non-parallel axes: the typical case, with curved creases
a     = [-1.7 -1.1;  1.9 -1.4;  0.2  1.9; -2.4  1.6];   % apices
w     = [ 0.25 0.10; -0.30 0.18;  0.05 -0.28; 0.22 -0.05];
alpha = [ 1.00        1.45        0.85        1.20];
rho   = [ 0.00       -0.35        0.25        0.10];
K     = size(a, 1);
eps0  = 0.05;                                   % apex smoothing

n  = 420;
xv = linspace(-5, 5, n); yv = linspace(-4.5, 4.5, n);
[X, Y] = meshgrid(xv, yv);
G = zeros(n, n, K);
for k = 1:K
    ux = X - a(k,1); uy = Y - a(k,2);
    G(:,:,k) = ux*w(k,1) + uy*w(k,2) ...
             + alpha(k)*sqrt(ux.^2 + uy.^2 + eps0^2) - rho(k);
end
[E, idx] = min(G, [], 3);                        % lower envelope and owning cone

% ---------------------------------------------------------------- figure 1
f1 = figure('position', [100 100 760 520]);
s = 6;                                           % mesh stride
surf(X(1:s:end,1:s:end), Y(1:s:end,1:s:end), E(1:s:end,1:s:end), ...
     'facecolor', [0.88 0.90 0.94], 'edgecolor', [0.35 0.35 0.40], ...
     'linewidth', 0.3, 'facealpha', 0.95);
hold on;
% creases: where the owning cone changes
[gx, gy] = gradient(double(idx));
crease = (abs(gx) + abs(gy)) > 0;
plot3(X(crease), Y(crease), E(crease) + 0.02, '.', ...
      'markersize', 2, 'color', [0.75 0.15 0.15]);
for k = 1:K
    plot3(a(k,1), a(k,2), min(min(E)) - 0.15, 'k.', 'markersize', 18);
end
view(-38, 26); axis tight; grid on;
xlabel('z_1'); ylabel('z_2'); zlabel('min_k (g_k(z) - \rho_k)');
if isoct                                        % Octave: page size in points with -S
    print(f1, 'cone_envelope.pdf', '-dpdf', '-S760,520');
else                                            % MATLAB: page size from PaperSize
    set(f1, 'PaperUnits', 'points', 'PaperSize', [760 520], 'PaperPosition', [0 0 760 520]);
    print(f1, 'cone_envelope.pdf', '-dpdf');
end

% ---------------------------------------------------------------- figure 2
f2 = figure('position', [100 100 620 560]);
imagesc(xv, yv, idx); set(gca, 'ydir', 'normal');
colormap([0.93 0.90 0.84; 0.85 0.90 0.93; 0.90 0.86 0.92; 0.87 0.93 0.88]);
hold on;
contour(X, Y, double(idx), 0.5:1:(K-0.5), 'linecolor', [0.15 0.15 0.15], ...
        'linewidth', 1.4);
% pick the pair of cells whose crease deviates most from a straight line, and
% draw that line for comparison
best = 0; bestP = [];
for p1 = 1:K
  for p2 = p1+1:K
    nb = false(size(idx));
    nb(:,1:end-1) = nb(:,1:end-1) | (idx(:,2:end)   == p2);
    nb(:,2:end)   = nb(:,2:end)   | (idx(:,1:end-1) == p2);
    nb(1:end-1,:) = nb(1:end-1,:) | (idx(2:end,:)   == p2);
    nb(2:end,:)   = nb(2:end,:)   | (idx(1:end-1,:) == p2);
    [iy, ix] = find((idx == p1) & nb);
    if numel(ix) < 30, continue; end
    P = [xv(ix)' yv(iy)'];
    d = P - mean(P); [~, S, V] = svd(d, 0);
    dev = S(2,2)/S(1,1);                    % the straightness ratio of Section 7.8
    if dev > best, best = dev; bestP = P; end
  end
end
if ~isempty(bestP)
    d = bestP - mean(bestP); [~, ~, V] = svd(d, 0);
    t = d*V(:,1); ends = mean(bestP) + [min(t); max(t)]*V(:,1)';
    plot(ends(:,1), ends(:,2), '--', 'color', [0.75 0.15 0.15], 'linewidth', 1.8);
    fprintf('most curved crease: straightness ratio %.4f\n', best);
end
for k = 1:K
    plot(a(k,1), a(k,2), 'k.', 'markersize', 18);
    text(a(k,1)+0.12, a(k,2)+0.18, sprintf('a_%d', k));
end
axis equal tight; xlabel('z_1'); ylabel('z_2');
if isoct
    print(f2, 'cone_cells.pdf', '-dpdf', '-S620,560');
else
    set(f2, 'PaperUnits', 'points', 'PaperSize', [620 560], 'PaperPosition', [0 0 620 560]);
    print(f2, 'cone_cells.pdf', '-dpdf');
end

disp('wrote cone_envelope.pdf and cone_cells.pdf');
close(f1); close(f2);
if ~isempty(oldtk), graphics_toolkit(oldtk); end
