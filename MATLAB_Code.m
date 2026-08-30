% =Graphene Sample biased at 100nA current, data is aquired for 32 min========================================================================
%  noise_psd_analysis_100na32minauto_logfit.m
%  1/f Noise PSD Analysis — Graphene / Resistor Sample
%
%  Pipeline:
%    1. Load raw X (in-phase) and Y (quadrature) lock-in data
%    2. 3-stage Kaiser FIR decimation  (1024 Hz -> 32 Hz)
%    3. Phase correction  (rotate XY so signal is in X)
%    4. Lock-in roll-off correction  (24 dB/oct, tau = 10 ms)
%    5. PSD estimation via pwelch / cpsd / mscohere
%    6. Johnson-Nyquist floor check  (S0 >= 4kTR)
%    7. SNR-based valid fitting region  (SNR > 3 contiguous from low-f)
%    8. Power-law fit on Sxx-Syy in valid region  -> alpha, A, f_cross
%    9. Two publication-quality log-log plots
%
%  Requires: Signal Processing Toolbox, DSP System Toolbox,
%            Curve Fitting Toolbox, Statistics & ML Toolbox
%
%  Author : Generated for graphene 1/f noise experiment
%  Data   : xdata100na32minauto, ydata100na32minauto  (1966080 samples, 1024 Hz)
% =========================================================================

clear; clc; close all;

%% =========================================================================
%  USER PARAMETERS  — edit these for each measurement
% =========================================================================
FS_RAW      = 1024;        % Raw sampling rate (Hz)
DEC_FACTOR  = 32;          % Total decimation factor
TAU         = 0.01;        % Lock-in time constant (s)
ROLL_ORDER  = 4;           % Lock-in roll-off order (24dB/oct = 4th order)
F_REF       = 227;         % Lock-in reference frequency (Hz)
T_SAMPLE    = 300;         % Sample temperature (K)
I_BIAS      = 100e-9;      % Bias current (A) — update per dataset
% R_SAMPLE computed from data after loading: R = mean(raw_X) / I_BIAS
SNR_THRESH  = 3;           % SNR threshold for valid fitting region
XFILE       = 'xdata100na32minauto';   % X channel data file
YFILE       = 'ydata100na32minauto';   % Y channel data file

% Welch PSD parameters
NFFT_SEGS   = 8;           % Number of segments for Welch (more = smoother PSD)
OVERLAP_PCT = 50;          % Overlap percentage for Welch

%% =========================================================================
%  STAGE 1 — Load raw data
% =========================================================================
fprintf('=== Loading data ===\n');
raw_X = load(XFILE);
raw_Y = load(YFILE);
raw_X = raw_X(:);
raw_Y = raw_Y(:);

N_raw = length(raw_X);
fprintf('  Loaded %d samples per channel\n', N_raw);
fprintf('  Duration: %.1f s (%.2f min) at %d Hz\n', ...
    N_raw/FS_RAW, N_raw/FS_RAW/60, FS_RAW);

% Compute sample resistance from Ohm's law: R = mean(V) / I_bias
% mean(raw_X) is the DC component of the lock-in X channel = V across sample
R_SAMPLE = mean(raw_X) / I_BIAS;
fprintf('  R_SAMPLE = mean(X)/I_bias = %.4e/%.4e = %.2f Ohm\n', ...
    mean(raw_X), I_BIAS, R_SAMPLE);

%% =========================================================================
%  STAGE 2 — 3-stage Kaiser FIR Decimation using DSP System Toolbox
%  Target: 1024 Hz -> 32 Hz  (factor 32 = 4 x 4 x 2)
% =========================================================================
fprintf('\n=== Decimation (x%d: %d Hz -> %d Hz) ===\n', ...
    DEC_FACTOR, FS_RAW, FS_RAW/DEC_FACTOR);

% Decompose decimation factor into 3 stages: 32 = 8 x 4 x 1  (or 4x4x2)
% Use 4 x 4 x 2 for balanced filter lengths
D1 = 4; D2 = 4; D3 = 2;
assert(D1*D2*D3 == DEC_FACTOR, 'Stage decimation factors must multiply to DEC_FACTOR');

% Design Kaiser FIR filters for each stage
% Passband edge: 0.48 x (Nyquist after decimation)
% Stopband attenuation: 100 dB for clean spectral estimation
ATTEN_DB = 100;
fs_s1 = FS_RAW;
fs_s2 = FS_RAW / D1;
fs_s3 = FS_RAW / (D1*D2);
fs_eff = FS_RAW / DEC_FACTOR;

% Normalized passband and stopband edges for each stage
% Passband: 0 to 0.45*Nyquist_out, Stopband: 0.5*Nyquist_out
Fp1 = 0.45 * (fs_eff);   % Hz — common passband for all stages (noise band of interest)
Fs1 = fs_s1/D1/2;        % Hz — stopband starts at new Nyquist after stage 1
Fp2 = 0.45 * fs_eff;
Fs2 = fs_s2/D2/2;
Fp3 = 0.45 * fs_eff;
Fs3 = fs_s3/D3/2;

% Design using kaiserord + fir1
[n1,Wn1,beta1,ftype1] = kaiserord([Fp1 Fs1],[1 0],[10^(-ATTEN_DB/20) 10^(-ATTEN_DB/20)/2],fs_s1);
if mod(n1,2)==0, n1=n1+1; end   % force odd length for linear phase
h1 = fir1(n1, Wn1, ftype1, kaiser(n1+1,beta1), 'noscale');

[n2,Wn2,beta2,ftype2] = kaiserord([Fp2 Fs2],[1 0],[10^(-ATTEN_DB/20) 10^(-ATTEN_DB/20)/2],fs_s2);
if mod(n2,2)==0, n2=n2+1; end
h2 = fir1(n2, Wn2, ftype2, kaiser(n2+1,beta2), 'noscale');

[n3,Wn3,beta3,ftype3] = kaiserord([Fp3 Fs3],[1 0],[10^(-ATTEN_DB/20) 10^(-ATTEN_DB/20)/2],fs_s3);
if mod(n3,2)==0, n3=n3+1; end
h3 = fir1(n3, Wn3, ftype3, kaiser(n3+1,beta3), 'noscale');

fprintf('  Stage 1 (x%d): %d taps, fs=%.0f Hz\n', D1, n1+1, fs_s1);
fprintf('  Stage 2 (x%d): %d taps, fs=%.0f Hz\n', D2, n2+1, fs_s2);
fprintf('  Stage 3 (x%d): %d taps, fs=%.0f Hz\n', D3, n3+1, fs_s3);

% Apply decimation using dsp.FIRDecimator for each stage
dec1 = dsp.FIRDecimator(D1, h1);
dec2 = dsp.FIRDecimator(D2, h2);
dec3 = dsp.FIRDecimator(D3, h3);

% Decimate X channel
tmpX1  = dec1(raw_X);
tmpX2  = dec2(tmpX1);
dec_X  = dec3(tmpX2);

% Reset filter states before processing Y
reset(dec1); reset(dec2); reset(dec3);

% Decimate Y channel
tmpY1  = dec1(raw_Y);
tmpY2  = dec2(tmpY1);
dec_Y  = dec3(tmpY2);

N_dec = length(dec_X);
fprintf('  Decimated: %d samples @ %.1f Hz\n', N_dec, fs_eff);
fprintf('  Nyquist after decimation: %.1f Hz\n', fs_eff/2);

%% =========================================================================
%  STAGE 3 — Phase correction
%  Rotate (X,Y) so that the carrier signal lies entirely in X channel
% =========================================================================
fprintf('\n=== Phase correction ===\n');
phi0 = angle(mean(dec_X + 1i*dec_Y));
fprintf('  phi0 = %.4f rad (%.2f deg)\n', phi0, rad2deg(phi0));

Xc =  dec_X * cos(phi0) + dec_Y * sin(phi0);   % corrected in-phase
Yc = -dec_X * sin(phi0) + dec_Y * cos(phi0);   % corrected quadrature

fprintf('  Mean after correction — Xc: %.4e V   Yc: %.4e V\n', ...
    mean(Xc), mean(Yc));

%% =========================================================================
%  STAGE 4 — PSD Estimation via Welch's method
%  pwelch for Sxx, Syy  |  cpsd for Sxy  |  mscohere for coherence
% =========================================================================
fprintf('\n=== PSD estimation (Welch) ===\n');

% Segment length: divide total samples by number of desired segments
% More segments -> smoother PSD but lower frequency resolution
seg_len  = floor(N_dec / NFFT_SEGS);
n_overlap = floor(seg_len * OVERLAP_PCT/100);
win       = hann(seg_len);

fprintf('  Segment length : %d samples\n', seg_len);
fprintf('  Overlap        : %d samples (%d%%)\n', n_overlap, OVERLAP_PCT);
fprintf('  Frequency res  : %.4f Hz\n', fs_eff/seg_len);

[Sxx, freq] = pwelch(Xc, win, n_overlap, seg_len, fs_eff, 'onesided');
[Syy, ~   ] = pwelch(Yc, win, n_overlap, seg_len, fs_eff, 'onesided');
[Sxy, ~   ] = cpsd(Xc, Yc, win, n_overlap, seg_len, fs_eff, 'onesided');
[Coh, ~   ] = mscohere(Xc, Yc, win, n_overlap, seg_len, fs_eff);

% Remove DC bin (f=0) — not meaningful for 1/f analysis
dc_mask = freq > 0;
freq = freq(dc_mask);
Sxx  = Sxx(dc_mask);
Syy  = Syy(dc_mask);
Sxy  = Sxy(dc_mask);
Coh  = Coh(dc_mask);

% Bin 1 is retained — log-space fit with bisquare robust weighting
% handles the DC leakage artifact in bin 1 without needing to remove it.

%% =========================================================================
%  STAGE 5 — Lock-in roll-off correction
%  H(f) = 1 / (1 + i*2*pi*f*tau)^roll_order
%  Correction factor = |H(f)|^{-2} = (1 + (2*pi*f*tau)^2)^roll_order
% =========================================================================
fprintf('\n=== Lock-in roll-off correction ===\n');
fprintf('  tau = %.3f s,  order = %d  (%.0f dB/oct)\n', ...
    TAU, ROLL_ORDER, 20*log10(2)*ROLL_ORDER);

corr_factor = (1 + (2*pi*freq*TAU).^2).^ROLL_ORDER;

Sxx_corr = Sxx .* corr_factor;
Syy_corr = Syy .* corr_factor;

% Correction magnitude at key frequencies
fprintf('  Correction at 1 Hz  : %.4f (%.2f dB)\n', ...
    interp1(freq, corr_factor, 1,  'linear','extrap'), ...
    10*log10(interp1(freq, corr_factor, 1, 'linear','extrap')));
fprintf('  Correction at 10 Hz : %.4f (%.2f dB)\n', ...
    interp1(freq, corr_factor, min(10,max(freq)), 'linear','extrap'), ...
    10*log10(interp1(freq, corr_factor, min(10,max(freq)), 'linear','extrap')));

%% =========================================================================
%  STAGE 6 — Noise floor and Johnson-Nyquist check
% =========================================================================
fprintf('\n=== Noise floor & Johnson-Nyquist check ===\n');

% S0: median of Syy in upper half of frequency range (cleanest white noise region)
high_f_mask = freq > 0.5 * max(freq);
S0          = median(Syy_corr(high_f_mask));
S0_mad      = 1.4826 * median(abs(Syy_corr(high_f_mask) - S0));  % robust std

fprintf('  S0 (median Syy, high-f)  : %.4e V²/Hz\n', S0);
fprintf('  S0 MAD uncertainty       : %.4e V²/Hz\n', S0_mad);

% Johnson-Nyquist floor
kB        = 1.38064852e-23;   % Boltzmann constant
johnson   = 4 * kB * T_SAMPLE * R_SAMPLE;
fprintf('  Johnson floor 4kTR       : %.4e V²/Hz  (R=%.0f Ohm, T=%.0f K)\n', ...
    johnson, R_SAMPLE, T_SAMPLE);

if S0 >= johnson
    fprintf('  CHECK PASSED: S0 >= 4kTR  (ratio = %.2f)\n', S0/johnson);
    johnson_ok = true;
else
    fprintf('  WARNING: S0 < 4kTR! Check gain calibration or R_SAMPLE value.\n');
    fprintf('  S0/4kTR = %.4f\n', S0/johnson);
    johnson_ok = false;
end

%% =========================================================================
%  STAGE 7 — Sample noise and valid fitting region
% =========================================================================
fprintf('\n=== Valid fitting region (SNR threshold = %.0f) ===\n', SNR_THRESH);

% Sample noise: Sxx - Syy (corrected)
Ssample = Sxx_corr - Syy_corr;

% SNR at each frequency bin
SNR = Ssample ./ Syy_corr;

% Find largest contiguous block starting from lowest frequency where SNR > threshold
above = SNR > SNR_THRESH;

% Walk from lowest frequency upward — stop at first break
valid_mask = false(size(above));
in_block   = false;
for k = 1:length(above)
    if above(k)
        valid_mask(k) = true;
        in_block = true;
    else
        if in_block
            break;   % stop at first break in contiguous block from low-f
        end
    end
end

if sum(valid_mask) < 3
    warning(['Valid fitting region has fewer than 3 points. ' ...
             'Try lowering SNR_THRESH or increasing acquisition time.']);
    % Fall back to all positive sample noise points
    valid_mask = Ssample > 0;
end

f_valid   = freq(valid_mask);
S_valid   = Ssample(valid_mask);

fprintf('  Valid region   : %.4f Hz — %.4f Hz  (%d bins)\n', ...
    f_valid(1), f_valid(end), sum(valid_mask));

%% =========================================================================
%  STAGE 8 — Power-law fit in LOG-LOG space
%
%  Model:   S(f) = A * f^{-alpha}
%  Taking log10 both sides:
%           log10(S) = log10(A) - alpha * log10(f)
%           Y        = C        + p(1)  * X
%  This is a straight line in log-log space: Y = p(1)*X + p(2)
%  where  p(1) = -alpha,  p(2) = log10(A)
%
%  Why log-space:
%    Linear-space fit minimises sum((S_data - S_fit)^2).
%    A point at 10^-12 contributes 10^8 times more than a point at 10^-16
%    so high bins dominate completely regardless of how many low bins exist.
%    Log-space fit minimises sum((log S_data - log S_fit)^2) — every
%    decade contributes equally, giving the correct slope and intercept.
% =========================================================================
fprintf('\n=== Power-law fit (log-log space) ===\n');

% Transform to log space — only positive valid points
log_f  = log10(f_valid);
log_S  = log10(S_valid);

% Linear fit in log space: log10(S) = p(1)*log10(f) + p(2)
% p(1) = -alpha,  p(2) = log10(A)
% Use Curve Fitting Toolbox fit() on log-transformed data for CI estimates
ft_log   = fittype('p1*x + p2', 'independent','x', ...
                   'coefficients',{'p1','p2'});
opts_log = fitoptions(ft_log);
opts_log.StartPoint = [-1.0, log10(median(S_valid))];  % alpha~1, A~median
opts_log.Lower      = [-3,  -Inf];    % alpha in [0, 3]
opts_log.Upper      = [ 0,   Inf];
opts_log.Robust     = 'Bisquare';     % robust to remaining outlier bins

fit_log    = fit(log_f, log_S, ft_log, opts_log);
ci_log     = confint(fit_log, 0.95);   % 95% CI on [p1, p2]

alpha_fit  = -fit_log.p1;             % slope = -alpha
A_fit      = 10^fit_log.p2;           % intercept -> A
alpha_ci   = [-ci_log(2,1), -ci_log(1,1)];   % flip sign for alpha
A_ci       = [10^ci_log(1,2), 10^ci_log(2,2)];

fprintf('  alpha = %.4f  [%.4f, %.4f]  (95%% CI)\n', ...
    alpha_fit, alpha_ci(1), alpha_ci(2));
fprintf('  A     = %.4e  [%.4e, %.4e]  V²/Hz  (at 1 Hz)\n', ...
    A_fit, A_ci(1), A_ci(2));

% Crossover frequency: A * f_cross^{-alpha} = S0
% => f_cross = (A/S0)^{1/alpha}
f_cross = (A_fit / S0)^(1/alpha_fit);
fprintf('  f_cross = %.4f Hz\n', f_cross);

% Sanity check — warn if f_cross outside measurement band
if f_cross > fs_eff/2
    fprintf('  NOTE: f_cross > Nyquist (%.1f Hz) — 1/f component crosses\n', fs_eff/2);
    fprintf('        background outside measurement band. Higher bias needed.\n');
elseif f_cross < freq(1)
    fprintf('  NOTE: f_cross < lowest bin (%.4f Hz) — crossover below\n', freq(1));
    fprintf('        measurement band. Longer acquisition needed.\n');
else
    fprintf('  f_cross is within measurement band — good.\n');
end

% Fitted curve over full frequency range for plot overlay
f_fit_line = logspace(log10(freq(1)), log10(freq(end)), 500)';
S_fit_line = A_fit * f_fit_line.^(-alpha_fit);

%% =========================================================================
%  STAGE 9 — PLOT 1: Sxx (green), Syy (red), Sxx-Syy (blue)
%            All axes log scale, horizontal dashed line at S0
% =========================================================================
fprintf('\n=== Plotting ===\n');

fig1 = figure('Name','PSD — Sxx, Syy, Sxx-Syy', ...
    'Position',[100 100 720 520], 'Color','w');

ax1 = axes(fig1);

% --- Sxx: total noise (green, open circles) ---
h_sxx = semilogy(ax1, freq, Sxx_corr, '-o', ...
    'Color',[0.0 0.6 0.0], ...
    'MarkerFaceColor','none', ...
    'MarkerEdgeColor',[0.0 0.6 0.0], ...
    'MarkerSize', 4, ...
    'LineWidth', 1.2, ...
    'DisplayName','S_{XX}  (Total noise)');
hold(ax1, 'on');

% --- Syy: background noise (red, filled triangles) ---
h_syy = semilogy(ax1, freq, Syy_corr, '-^', ...
    'Color',[0.85 0.1 0.1], ...
    'MarkerFaceColor',[0.85 0.1 0.1], ...
    'MarkerEdgeColor',[0.85 0.1 0.1], ...
    'MarkerSize', 4, ...
    'LineWidth', 1.2, ...
    'DisplayName','S_{YY}  (Background noise)');

% --- Sxx - Syy: sample noise (blue, filled circles) ---
% Only plot where Ssample > 0 (log scale can't show negatives)
pos_mask  = Ssample > 0;
h_samp = semilogy(ax1, freq(pos_mask), Ssample(pos_mask), '-o', ...
    'Color',[0.1 0.35 0.85], ...
    'MarkerFaceColor',[0.1 0.35 0.85], ...
    'MarkerEdgeColor',[0.1 0.35 0.85], ...
    'MarkerSize', 4, ...
    'LineWidth', 1.2, ...
    'DisplayName','S_{XX} - S_{YY}  (Sample noise)');

% --- S0 horizontal dashed line ---
h_s0 = yline(ax1, S0, '--', ...
    'Color',[0.5 0.5 0.5], ...
    'LineWidth', 1.5, ...
    'DisplayName', sprintf('S_0 = %.2e V^2/Hz  (noise floor)', S0));

% --- Johnson floor dashed line ---
h_jn = yline(ax1, johnson, ':', ...
    'Color',[0.7 0.4 0.0], ...
    'LineWidth', 1.5, ...
    'DisplayName', sprintf('4k_BTR = %.2e V^2/Hz  (Johnson floor)', johnson));

% --- Johnson check annotation ---
if johnson_ok
    jstr = sprintf('S_0 / 4k_BTR = %.2f  ✓', S0/johnson);
    jcol = [0.0 0.50 0.0];
else
    jstr = sprintf('S_0 / 4k_BTR = %.2f  ✗  Check calibration!', S0/johnson);
    jcol = [0.8 0.0 0.0];
end
text(ax1, min(freq)*1.5, S0*3, jstr, ...
    'FontSize',9, 'Color',jcol, 'FontWeight','bold');

hold(ax1, 'off');

% Axes formatting
set(ax1, 'XScale','log', 'YScale','log');
xlabel(ax1, 'f  (Hz)',       'FontSize',13, 'FontWeight','bold');
ylabel(ax1, 'S_V(f)  (V^2/Hz)', 'FontSize',13, 'FontWeight','bold');
title(ax1,  'Power Spectral Density — 1/f Noise Measurement', ...
    'FontSize',13, 'FontWeight','bold');
legend(ax1, [h_sxx h_syy h_samp h_s0 h_jn], ...
    'Location','southwest', 'FontSize',9, 'Box','on');
grid(ax1, 'on');
ax1.GridAlpha       = 0.3;
ax1.MinorGridAlpha  = 0.15;
ax1.GridLineStyle   = ':';
ax1.Box             = 'on';
ax1.TickDir         = 'out';
ax1.FontSize        = 11;

% Tight x-limits
xlim(ax1, [min(freq)*0.8, max(freq)*1.2]);

%% =========================================================================
%  STAGE 10 — PLOT 2: Power-law fit on Sxx-Syy in valid region
%             Dashed fit line, alpha and A annotated, f_cross marked
% =========================================================================
fig2 = figure('Name','Sample Noise — Power-law fit', ...
    'Position',[840 100 720 520], 'Color','w');

ax2 = axes(fig2);

% --- Full sample noise (blue, faded, for context) ---
semilogy(ax2, freq(pos_mask), Ssample(pos_mask), 'o', ...
    'Color',[0.6 0.75 0.95], ...
    'MarkerFaceColor',[0.6 0.75 0.95], ...
    'MarkerEdgeColor',[0.6 0.75 0.95], ...
    'MarkerSize', 3.5, ...
    'DisplayName','S_{XX} - S_{YY}  (all bins)');
hold(ax2, 'on');

% --- Valid region (solid blue, prominent) ---
semilogy(ax2, f_valid, S_valid, 'o', ...
    'Color',[0.1 0.35 0.85], ...
    'MarkerFaceColor',[0.1 0.35 0.85], ...
    'MarkerEdgeColor',[0.1 0.35 0.85], ...
    'MarkerSize', 5, ...
    'DisplayName', sprintf('Fit region  (SNR > %g)', SNR_THRESH));

% --- Power-law fit: dashed line ---
semilogy(ax2, f_fit_line, S_fit_line, '--', ...
    'Color',[0.85 0.1 0.1], ...
    'LineWidth', 2.0, ...
    'DisplayName', sprintf('Fit: A·f^{-\\alpha}  (\\alpha = %.3f)', alpha_fit));

% --- S0 noise floor dashed line ---
yline(ax2, S0, '--', ...
    'Color',[0.5 0.5 0.5], ...
    'LineWidth', 1.5, ...
    'DisplayName', sprintf('S_0 = %.2e V^2/Hz', S0));

% --- Crossover frequency vertical dashed line ---
xline(ax2, f_cross, '--', ...
    'Color',[0.6 0.1 0.6], ...
    'LineWidth', 1.5, ...
    'DisplayName', sprintf('f_{cross} = %.3f Hz', f_cross));

% Force log x-axis immediately after xline — xline resets XScale to
% linear in some MATLAB versions so we re-apply it right here
ax2.XScale = 'log';

% --- Confidence interval shading ---
% Use ci_log from the log-space fit
S_lo    = 10^ci_log(1,2) * f_fit_line.^(-(-ci_log(2,1)));
S_hi    = 10^ci_log(2,2) * f_fit_line.^(-(-ci_log(1,1)));
% Clamp to positive values before fill (log axis cannot show negatives)
S_lo    = max(S_lo, min(S_valid)*0.01);
S_hi    = max(S_hi, min(S_valid)*0.01);
f_shade = [f_fit_line; flipud(f_fit_line)];
S_shade = [S_lo;       flipud(S_hi)];
fill(ax2, f_shade, S_shade, [0.85 0.1 0.1], ...
    'FaceAlpha',0.12, 'EdgeColor','none', ...
    'HandleVisibility','off');

hold(ax2, 'off');

% Axes formatting — set BEFORE annotation so text units are correct
set(ax2, 'XScale','log', 'YScale','log');
xlabel(ax2, 'f  (Hz)',                    'FontSize',13, 'FontWeight','bold');
ylabel(ax2, 'S_{XX} - S_{YY}  (V^2/Hz)', 'FontSize',13, 'FontWeight','bold');
title(ax2,  'Sample Noise — Power-law Fit  (valid SNR region)', ...
    'FontSize',13, 'FontWeight','bold');
legend(ax2, 'Location','southwest', 'FontSize',9, 'Box','on');
grid(ax2, 'on');
ax2.GridAlpha       = 0.3;
ax2.MinorGridAlpha  = 0.15;
ax2.GridLineStyle   = ':';
ax2.Box             = 'on';
ax2.TickDir         = 'out';
ax2.FontSize        = 11;
xlim(ax2, [min(freq)*0.8, max(freq)*1.2]);
drawnow;   % flush rendering so axis limits are finalised before text placement

% --- Annotation box: alpha, A, f_cross ---
% Place in upper-right of the data cloud — use axis limits for safe positioning
xl = xlim(ax2);
yl = ylim(ax2);
ann_x = 10^(log10(xl(1)) + 0.55*(log10(xl(2))-log10(xl(1))));  % 55% from left
ann_y = 10^(log10(yl(2)) - 0.05*(log10(yl(2))-log10(yl(1))));  % 5% from top

annot_str = sprintf( ...
    '\\alpha = %.3f  [%.3f, %.3f]\nA = %.3e V^2/Hz\nf_{cross} = %.3f Hz', ...
    alpha_fit, alpha_ci(1), alpha_ci(2), A_fit, f_cross);

text(ax2, ann_x, ann_y, annot_str, ...
    'FontSize', 10, ...
    'FontName', 'Courier New', ...
    'BackgroundColor', [0.97 0.97 0.97], ...
    'EdgeColor',       [0.70 0.70 0.70], ...
    'Margin',          5, ...
    'VerticalAlignment',   'top', ...
    'HorizontalAlignment', 'left');

%% =========================================================================
%  STAGE 11 — Print summary
% =========================================================================
fprintf('\n========================================\n');
fprintf('  RESULTS SUMMARY\n');
fprintf('========================================\n');
fprintf('  Effective fs         : %.2f Hz\n',     fs_eff);
fprintf('  Nyquist              : %.2f Hz\n',     fs_eff/2);
fprintf('  Freq range           : %.4f — %.4f Hz\n', freq(1), freq(end));
fprintf('  PSD bins             : %d\n',           length(freq));
fprintf('\n');
fprintf('  Noise floor S0       : %.4e V²/Hz\n',  S0);
fprintf('  Johnson floor 4kTR   : %.4e V²/Hz\n',  johnson);
fprintf('  Johnson ratio S0/4kTR: %.4f  %s\n',    S0/johnson, ...
    ternary_str(johnson_ok,'PASS','FAIL'));
fprintf('\n');
fprintf('  Valid fit region     : %.4f — %.4f Hz  (%d bins)\n', ...
    f_valid(1), f_valid(end), length(f_valid));
fprintf('  alpha (1/f slope)    : %.4f  [%.4f, %.4f]  (95%% CI)\n', ...
    alpha_fit, alpha_ci(1), alpha_ci(2));
fprintf('  A  (at 1 Hz)         : %.4e  [%.4e, %.4e]  V²/Hz\n', ...
    A_fit, A_ci(1), A_ci(2));
fprintf('  Crossover freq       : %.4f Hz\n',     f_cross);
fprintf('========================================\n');

%% =========================================================================
%  Helper function
% =========================================================================
function s = ternary_str(cond, a, b)
    if cond, s = a; else, s = b; end
end
