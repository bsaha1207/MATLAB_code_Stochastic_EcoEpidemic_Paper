% --------- LHS parameter sampling with truncated-Gamma for beta_ij and alpha_i ---------
clc; clear; rng(42);

N = 10000;  % number of rows (samples)

% ---- Parameter names (51 total) ----
names = { ...
  'D1','D2','D3', ...
  'V1','V2','V3', ...
  'c1','c2','c3', ...
  'b11','b12','b13','b21','b22','b23','b31','b32','b33', ...
  'beta_11','beta_12','beta_13','beta_21','beta_22','beta_23','beta_31','beta_32','beta_33', ...
  'alpha_1','alpha_2','alpha_3', ...
  'gamma_1','gamma_2','gamma_3', ...
  'mu_1','mu_2','mu_3', ...
  'rho_1','rho_2','rho_3', ...
  'delta', ...
  'sigma_1','sigma_2','sigma_3', ...
  'sigma_E', ...
  'gamma_jump1','gamma_jump2','gamma_jump3', ...
  'lambda_I1','lambda_I2','lambda_I3', ...
  'lambda_E'};

% ---- Lower/Upper bounds (match order exactly) ----
lo = [ ...
  0.005, 0.002, 0.001, ...                            % D1..D3
  0.004, 0.002, 0.001, ...                            % V1..V3
  82.13, 68.57, 144.32, ...                           % c1..c3
  43.30, 10.82, 28.01, 54.12, 14.43, 0.0145, 75.77, 39.69, 28.86, ...  % b_ij
  0.06,  0.01,  0.01,  0.01,  0.06,  0.01,  0.01,  0.01,  0.06, ...   % beta_ij
  3e-6,  3e-6,  3e-6,  ...                              % alpha_i
  0.75,  0.75,  0.75, ...                              % gamma_i
  0.0,   0.0,   0.0,  ...                              % mu_i
  1e1,   1e1,   1e1,  ...                              % rho_i  (log-uniform)
  2.0, ...                                             % delta
  0.1,   0.1,   0.1, ...                               % sigma_i
  0.1, ...                                             % sigma_E
  0.1,   0.1,   0.1, ...                               % gamma_jump i
  0.1,   0.1,   0.1, ...                               % lambda_I i
  0.1];                                                % lambda_E

hi = [ ...
  0.5,  0.2,  0.1, ...                                % D1..D3
  0.4,  0.2,  0.1, ...                                % V1..V3
  99.97, 83.47, 175.68, ...                           % c1..c3
  52.70, 13.18, 34.09, 65.88, 17.57, 0.0177, 92.23, 48.31, 35.14, ...  % b_ij
  1.52,  1.5,   1.5,   1.5,   1.5,   1.5,   1.5,   1.5,   1.52, ...    % beta_ij
  3e-2,  3e-2,  3e-2,  ...                              % alpha_i
  1.08,  1.08,  1.08, ...                              % gamma_i
  0.3,   0.3,   0.3,  ...                              % mu_i
  1e3,   1e3,   1e3,  ...                              % rho_i (log-uniform)
  10.0, ...                                            % delta
  1.2,   1.2,   1.2, ...                               % sigma_i
  1.5, ...                                             % sigma_E
  1.20,  1.20,  1.20, ...                              % gamma_jump i
  1.5,   1.5,   1.5, ...                               % lambda_I i
  1.2];                                                % lambda_E

% ---- Sanity checks ----
d = numel(names);
assert(numel(lo)==d && numel(hi)==d, 'lo/hi must match number of names.');
assert(all(hi > lo), 'All hi must be greater than lo.');

% ---- Distribution flags/indices ----
isLog = false(1,d);
% Keep rho_* as log-uniform (order-of-magnitude variability)
isLog(strcmp(names,'rho_1')) = true;
isLog(strcmp(names,'rho_2')) = true;
isLog(strcmp(names,'rho_3')) = true;

beta_idx  = ismember(names, {'beta_11','beta_12','beta_13', ...
                             'beta_21','beta_22','beta_23', ...
                             'beta_31','beta_32','beta_33'});

alpha_idx = ismember(names, {'alpha_1','alpha_2','alpha_3'});

% By default, map linearly; but exclude columns we will overwrite via Gamma
lin = ~isLog;
lin(beta_idx)  = false;   % will fill via truncated-Gamma
lin(alpha_idx) = false;   % will fill via truncated-Gamma

% ---- LHS uniforms (no maximin—fast & scalable) ----
U = lhsdesign(N, d);   % N-by-d in (0,1)

% ---- Allocate ----
X = zeros(N,d);

% ---- Linear-uniform columns (not log, not beta/alpha)
if any(lin)
    X(:,lin) = U(:,lin) .* (hi(lin) - lo(lin)) + lo(lin);
end

% ---- Log-uniform columns (here: rho_i)
if any(isLog)
    lo10 = log10(lo(isLog));
    hi10 = log10(hi(isLog));
    X(:,isLog) = 10.^( U(:,isLog) .* (hi10 - lo10) + lo10 );
end

% ---- Truncated-Gamma for beta_ij (CDF rescaling)
% Choose Gamma parameters for betas (tune as needed)
k_beta = 2;          % shape
th_beta = 0.2;      % scale (mean = 0.30)

if any(beta_idx)
    beta_cols = find(beta_idx);
    for t = 1:numel(beta_cols)
        j = beta_cols(t);
        a = lo(j); b = hi(j);
        Fa = cdf('Gamma', a, k_beta, th_beta);
        Fb = cdf('Gamma', b, k_beta, th_beta);
        Utr = U(:,j) .* (Fb - Fa) + Fa;            % map to [F(a), F(b)]
        X(:,j) = icdf('Gamma', Utr, k_beta, th_beta);
    end
end

% ---- Truncated-Gamma for alpha_i (CDF rescaling)
% Choose Gamma parameters for alphas so mean ~1e-3
k_alpha = 2;         % shape
th_alpha = 5e-4;     % scale (mean = 1e-3)

if any(alpha_idx)
    alpha_cols = find(alpha_idx);
    for t = 1:numel(alpha_cols)
        j = alpha_cols(t);
        a = lo(j); b = hi(j);
        Fa = cdf('Gamma', a, k_alpha, th_alpha);
        Fb = cdf('Gamma', b, k_alpha, th_alpha);
        Utr = U(:,j) .* (Fb - Fa) + Fa;            % map to [F(a), F(b)]
        X(:,j) = icdf('Gamma', Utr, k_alpha, th_alpha);
    end
end

% ---- Save
T = array2table(X,'VariableNames',names);
writetable(T, 'parameter_samples_10000_new.xlsx');

fprintf('Done. Wrote parameter_samples_10000_new.xlsx with %d rows and %d columns.\n', N, d);

%%

% ---------- Load ----------
T = readtable('parameter_samples_10000_new.xlsx');   % new file with per-species columns
N = height(T);
n = 3;

% ---------- Ecological block ----------
c  = [T.c1, T.c2, T.c3];
B11=T.b11; B12=T.b12; B13=T.b13;
B21=T.b21; B22=T.b22; B23=T.b23;
B31=T.b31; B32=T.b32; B33=T.b33;

% ---------- Preallocate ----------
R0 = nan(N,1);

for r = 1:N
    % --- B and c for this row ---
    Br = [B11(r) B12(r) B13(r);
          B21(r) B22(r) B23(r);
          B31(r) B32(r) B33(r)];
    cr = c(r,:).';
    Sstar = Br \ cr;                       % 3x1 (S^* = B^{-1} c)

    % ---------- Disease params (species-specific) ----------
    alpha_r = [T.alpha_1(r); T.alpha_2(r); T.alpha_3(r)];   % 3x1
    gamma_r = [T.gamma_1(r); T.gamma_2(r); T.gamma_3(r)];   % 3x1
    mu_r    = [T.mu_1(r);    T.mu_2(r);    T.mu_3(r)   ];   % 3x1
    rho_r   = [T.rho_1(r);   T.rho_2(r);   T.rho_3(r)  ];   % 3x1
    sigma_r = [T.sigma_1(r); T.sigma_2(r); T.sigma_3(r)];   % 3x1

    % Environment noise (scalar)
    sigmaE  = T.sigma_E(r);

    % Jump parameters (compound Poisson, constant amplitudes)
    gjump   = [T.gamma_jump1(r); T.gamma_jump2(r); T.gamma_jump3(r)];   % 3x1
    lambdaI = [T.lambda_I1(r);   T.lambda_I2(r);   T.lambda_I3(r)  ];   % 3x1
    lambdaE = T.lambda_E(r);                                            % scalar

    % Option for environment jump amplitude: use mean of species amplitudes
    gE = mean(gjump);

    % Jump penalties: J = lambda * (gamma - log(1+gamma))
    jump_core_I = gjump - log(1 + gjump);     % 3x1, >=0 for gjump>-1
    JI = lambdaI .* jump_core_I;              % 3x1
    JE = lambdaE * (gE - log(1 + gE));        % scalar

    % ---------- Beta matrix (3x3) from nine columns ----------
    beta = [T.beta_11(r) T.beta_12(r) T.beta_13(r);
            T.beta_21(r) T.beta_22(r) T.beta_23(r);
            T.beta_31(r) T.beta_32(r) T.beta_33(r)];

   %beta = [T.beta_11(r) 0 0;
           % 0 T.beta_22(r) 0;
           % 0 0 T.beta_33(r)];


    % ---------- Tilde terms (Brownian + jumps; periodic/Neumann) ----------
    nu   = gamma_r + mu_r;                    % ν_i
    tnu  = nu + 0.5*(sigma_r.^2) + JI;        % \tilde{\nu}_i (3x1)
    tdel = T.delta(r) + 0.5*(sigmaE^2) + JE;  % \tilde{\delta}  (scalar)

    % Safety: avoid divide-by-zero (very unlikely but robust to NaNs/Infs)
    if any(~isfinite(tnu)) || ~isfinite(tdel) || any(tnu<=0) || tdel<=0
        R0(r) = NaN;
        continue
    end

    % ---------- K_eff and spectral radius ----------
    K  = diag(Sstar) * ( beta + (alpha_r * (rho_r.'))/tdel ) * diag(1./tnu);
    ev = eig(K);
    R0(r) = max(abs(ev));
end

% ---------- Save back ----------
T.R0 = R0;
writetable(T, 'parameter_samples_10000_new.xlsx', 'Sheet', 1);

%%

% Load cleaned dataset
T = readtable('goodoutputdataset.xlsx');  % or parameter_samples_50000.xlsx

% 1) Remove anomalies (if not already removed)
%T = T(T.R0 <= 10, :);

% 2) Identify numeric columns (includes R0 if numeric)
vars = T.Properties.VariableNames;
numVarNames = {};
summaryStats = [];   % rows = variables, cols = stats

for j = 1:numel(vars)
    x = T.(vars{j});
    if isnumeric(x)
        % Basic stats with NaN safety
        m   = mean(x, 'omitnan');
        sd  = std(x,  'omitnan');
        mn  = min(x, [], 'omitnan');
        p25 = prctile(x, 25);
        med = median(x, 'omitnan');
        p75 = prctile(x, 75);
        mx  = max(x, [], 'omitnan');
        n   = sum(~isnan(x));  % count of non-NaN entries

        summaryStats = [summaryStats; [m sd mn p25 med p75 mx n]];
        numVarNames{end+1} = vars{j}; %#ok<SAGROW>
    end
end

% 3) Convert to table
statsTable = array2table(summaryStats, ...
    'VariableNames', {'Mean','StdDev','Min','P25','Median','P75','Max','N'}, ...
    'RowNames', numVarNames);

% 4) Show in Command Window
%disp('=== Summary statistics (after filtering R0 <= 10) ===');
disp(statsTable);

% 5) Save to Excel for supplement
writetable(statsTable, 'goodoutput_summary_stats.xlsx', 'WriteRowNames', true);


%%

% Load your table
T = readtable('parameter_samples_10000_new.xlsx');   % or parameter_samples_50000.xlsx

% (Optional) keep only sensible rows if you haven't already
% T = T(T.R0 <= 10, :);

% Create binary dependent variable: 1 if R0 > 1, else 0
R0_bin = double(T.R0 > 1);

% Insert it right after the R0 column (name it whatever you like)
T = addvars(T, R0_bin, 'After', 'R0', 'NewVariableNames', 'R0_binary');

% Quick sanity check in Command Window
n1 = sum(T.R0_binary == 1);
n0 = sum(T.R0_binary == 0);
fprintf('R0_binary counts -> 1: %d, 0: %d (total: %d)\n', n1, n0, height(T));

% Save back to Excel (overwrites the file with the new column)
writetable(T, 'parameter_samples_10000_new.xlsx', 'Sheet', 1);

