% crease_control.m
%
% Synthetic control for Section 6. Runs in MATLAB and Octave, no data needed.
% A line of 257 points is evaluated with the estimator of src/crease_lines.py:
% D(h) = max |L(i+1) - 2 L(i) + L(i-1)| / h^2 over the central half of the line, at the
% spacings h, 2h, 4h, and gamma is minus the slope of log D against log h.
%
%   (1) a parabola plus a random walk in the slope (a dense sum of small kinks of
%       varying sign), for several weights r of the random walk;
%   (2) a single crease (a jump of the slope) at a random position between grid points;
%   (3) the bound: a crease of size J is never larger than 4 h D(h).
%
% Prints the median and the 5-95% range of gamma for each case.

function crease_control()
rng(0);                                    % MATLAB; Octave accepts it as well
P = 257; h = 1; sub = 16; n = (P-1)*sub; reps = 400;
cases = {'smooth (curvature only)', 0, 0; 'random kinks, weak', 0.3, 0;
         'random kinks, comparable', 1, 0; 'random kinks, strong', 3, 0;
         'random kinks only', 1e3, 0};
for c = 1:size(cases, 1)
    g = zeros(reps, 1);
    for k = 1:reps, g(k) = gamma1d(line(cases{c,2}, 0, 0), h); end
    report(cases{c,1}, g);
end
g = zeros(reps, 1);
for k = 1:reps, g(k) = gamma1d(line(0, 20, 4*rand), h); end
report('single crease, random position', g);
ok = 0; N = 4000;
for k = 1:N
    J = 0.5 + 49.5*rand;  L = line(5*rand, J, rand - 0.5);
    lo = floor(P*0.25); hi = floor(P*0.75);
    d2 = abs(L(3:end) - 2*L(2:end-1) + L(1:end-2));  D = max(d2(lo+1:hi)) / h^2;
    ok = ok + (J <= 4*h*D);
end
fprintf('crease size <= 4 h D(h) in %d of %d synthetic lines\n', ok, N);

    function L = line(r, jump, off)
        t = (0:n)' * (h/sub);
        slope = [0; cumsum(randn(n, 1) * sqrt(h/sub) * r)] + (t - mean(t)) ...
              + jump * (t > mean(t) + off*h);
        L = [0; cumsum(0.5*(slope(2:end) + slope(1:end-1)) * (h/sub))];
        L = L(1:sub:end);
    end
end

function g = gamma1d(L, h0)
P = numel(L); lo = floor(P*0.25); hi = floor(P*0.75);
s = [1 2 4]; D = zeros(1, 3);
for k = 1:3
    S = L(1:s(k):end); a = floor(lo/s(k)); b = max(floor(hi/s(k)), a + 3);
    d2 = abs(S(3:end) - 2*S(2:end-1) + S(1:end-2));
    D(k) = max(d2(a+1:min(b, numel(d2)))) / (h0*s(k))^2;
end
p = polyfit(log(h0*s), log(D), 1); g = -p(1);
end

function report(name, g)
q = sort(g); n = numel(q);
fprintf('%-32s gamma median %.2f  5-95%% %.2f..%.2f\n', name, median(q), ...
        q(max(1, round(0.05*n))), q(round(0.95*n)));
end
