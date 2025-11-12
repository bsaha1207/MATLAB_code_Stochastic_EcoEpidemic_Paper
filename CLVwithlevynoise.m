clc; clear; close all;

% --- Period scaling
T_current = 2.85;     % approx current period
T_target  = 12;       % desired period
tau = T_current / T_target;

% --- Class 27 parameters
eta_c = 2/53 * (421 + sqrt(13471));
nu_c  = 64 - 3*eta_c;

eta = eta_c + 1/20;
nu  = nu_c + 85/1e6;

A = tau * [48, 12, eta;
           60, 16, nu;
           84, 44, 32];
r = sum(A, 2);

% --- Time grids: pre (0->1500) and post (1500->2500)
dt        = 0.002;
t_pre     = [0, 1500];
t_post    = [1500, 2500];
t_perturb = t_pre(2);

N_pre   = round((t_pre(2)  - t_pre(1))  / dt);
N_post  = round((t_post(2) - t_post(1)) / dt);

t1 = linspace(t_pre(1),  t_pre(2),  N_pre+1);       % includes 0
t2 = linspace(t_post(1), t_post(2), N_post+1);      % includes 1500
t_full = [t1, t2(2:end)];                           % avoid duplicate at 1500

% --- Initial conditions
Y01 = [1.08; 1.05; 1.01];      % Set 1 baseline
Y02 = [1.0300; 1.0000; 1.0200];% Set 2 baseline

X1 = zeros(3, N_pre+1);  X1(:,1) = Y01;   % Set 1 pre
X2 = zeros(3, N_pre+1);  X2(:,1) = Y02;   % Set 2 pre

% --- Noise: same path (per phase) for both sets to isolate model effects
rng(1);
dW_pre  = sqrt(dt) * randn(3, N_pre);    % 3 x N_pre
dW_post = sqrt(dt) * randn(3, N_post);   % 3 x N_post

% --- Sigma (scalar or 3x1 vector) — keep your values
sigma1_pre  = 0.06;         % Set 1 pre
sigma2_pre  = 0.20;         % Set 2 pre
sigma1_post = 0.2 * sigma1_pre;   % keep as in your file
sigma2_post = 0.6 * sigma2_pre;

% ========================= Lévy Jump Settings (SAME as big SDE) =================
% Active windows
tJ0_pre  = 1500;    tJ1_pre  = 1530;      % pre-event window
tJ0_post = 1650;  tJ1_post = 1680;    % post-event window
% Intensities and amplitude std
lambdaJ_pre  = 0.20;  aJ_pre  = 0.20;   % small jump regime
lambdaJ_post = 0.60;  aJ_post = 0.80;   % large jump regime
% Target indices (here: all three species totals)
jump_idx_pre  = 1:3;
jump_idx_post = 1:3;

% --- === PHASE 1: Euler–Maruyama BEFORE perturbation ===
for i = 1:N_pre
    t_now = t1(i);

    % ---- Set 1 ----
    Xi   = X1(:, i);
    drift = Xi .* (r - A * Xi);
    diffusion = sigma1_pre * Xi;
    X_new = Xi + drift * dt + diffusion .* dW_pre(:, i);
    % Lévy jumps (multiplicative, log-compensated)
    X_new = levy_jump_update_vec(X_new, t_now, lambdaJ_pre, aJ_pre, dt, jump_idx_pre, tJ0_pre, tJ1_pre);
    X1(:, i+1) = max(X_new, 0);   % nonnegativity

    % ---- Set 2 (reuse SAME dW_pre(:,i)) ----
    Xj   = X2(:, i);
    drift = Xj .* (r - A * Xj);
    diffusion = sigma2_pre * Xj;
    X_new2 = Xj + drift * dt + diffusion .* dW_pre(:, i);
    X_new2 = levy_jump_update_vec(X_new2, t_now, lambdaJ_pre, aJ_pre, dt, jump_idx_pre, tJ0_pre, tJ1_pre);
    X2(:, i+1) = max(X_new2, 0);
end

X0_perturbed1 = X1(:, end);
X0_perturbed2 = X2(:, end);

% --- Apply perturbation to A and r (exactly your recipe)
Apert = A;
Apert(1,1) = A(1,1) - 0.08 * r(2);
Apert(2,1) = A(2,1) + 0.08 * r(1);

rpert = r;
rpert(1) = r(1) - 0.08 * A(2,1);
rpert(2) = r(2) - 0.08 * A(1,1);

% --- === PHASE 2: Euler–Maruyama AFTER perturbation ===
X1_post = zeros(3, N_post+1);   X1_post(:,1) = X0_perturbed1;
X2_post = zeros(3, N_post+1);   X2_post(:,1) = X0_perturbed2;

for k = 1:N_post
    t_now = t2(k);

    % ---- Set 1 ----
    Xi   = X1_post(:, k);
    drift = Xi .* (rpert - Apert * Xi);
    diffusion = sigma1_post * Xi;
    X_new = Xi + drift * dt + diffusion .* dW_post(:, k);
    X_new = levy_jump_update_vec(X_new, t_now, lambdaJ_post, aJ_post, dt, jump_idx_post, tJ0_post, tJ1_post);
    X1_post(:, k+1) = max(X_new, 0);

    % ---- Set 2 (reuse SAME dW_post(:,k)) ----
    Xj   = X2_post(:, k);
    drift = Xj .* (rpert - Apert * Xj);
    diffusion =  sigma2_post * Xj;           % keep your set-2 sigma here
    X_new2 = Xj + drift * dt + diffusion .* dW_post(:, k);
    X_new2 = levy_jump_update_vec(X_new2, t_now, lambdaJ_post, aJ_post, dt, jump_idx_post, tJ0_post, tJ1_post);
    X2_post(:, k+1) = max(X_new2, 0);
end

% --- Merge trajectories (avoid duplicate t = 1500)
X_full1 = [X1,        X1_post(:, 2:end)];   % Set 1
X_full2 = [X2,        X2_post(:, 2:end)];   % Set 2

% --- Plotting (unchanged)
c1 = [0 0.4470 0.7410];      % susceptible set color
c2 = [0.4940 0.1840 0.5560]; % disease-carrying set color

% Phase portrait (after perturbation only)
figure('Color','w'); hold on; grid on; box on
plot3(X1_post(1,1:10:end), X1_post(2,1:10:end), X1_post(3,1:10:end), '-',  ...
      'Color',c1,'LineWidth',1.2, 'DisplayName','Susceptible (after perturbation)');
plot3(X2_post(1,1:10:end), X2_post(2,1:10:end), X2_post(3,1:10:end), '--', ...
      'Color',c2,'LineWidth',1.2, 'DisplayName','Disease Carrying (after perturbation)');
xlabel('X_1'); ylabel('X_2'); zlabel('X_3');
title('Stochastic Phase Portrait (Post-Perturbation)'); legend('Location','best');
view(135,25);

% Phase portrait (full pre + post)
figure('Color','w'); hold on; grid on; box on
plot3(X_full1(1,1:10:end), X_full1(2,1:10:end), X_full1(3,1:10:end), '-',  ...
      'Color',c1,'LineWidth',1.2, 'DisplayName','Susceptible (pre+post)');
plot3(X_full2(1,1:10:end), X_full2(2,1:10:end), X_full2(3,1:10:end), '--', ...
      'Color',c2,'LineWidth',1.2, 'DisplayName','Disease Carrying (pre+post)');
xlabel('X_1'); ylabel('X_2'); zlabel('X_3');
title('Stochastic Phase Portrait (Baseline \rightarrow Perturbation)'); legend('Location','best');
view(135,25); axis tight

% Species time series (pre only)
labels = {'X_1','X_2','X_3'};
for k = 1:3
    figure('Color','w'); hold on; grid on; box on
    plot(t1, X1(k,:), 'b',  'LineWidth', 1.2, 'DisplayName','Set 1 (pre)');
    plot(t1, X2(k,:), 'b--', 'LineWidth', 1.2, 'DisplayName','Set 2 (pre)');
    xline(t_perturb, '--k', 'LineWidth', 1.0, 'DisplayName','perturb');
    xlabel('Time (months)'); ylabel(labels{k});
    title(sprintf('Pre-Perturbation: %s', labels{k}));
    legend('Location','best'); ylim([0 inf]);
end

% Species time series (post only)
for k = 1:3
    figure('Color','w'); hold on; grid on; box on
    plot(t2, X1_post(k,:), 'r',  'LineWidth', 1.2, 'DisplayName','Set 1 (post)');
    plot(t2, X2_post(k,:), 'r--', 'LineWidth', 1.2, 'DisplayName','Set 2 (post)');
    xline(t_perturb, '--k', 'LineWidth', 1.0, 'DisplayName','perturb start');
    xlabel('Time (months)'); ylabel(labels{k});
    title(sprintf('Post-Perturbation: %s', labels{k}));
    legend('Location','best'); ylim([0 inf]);
end

% Species time series (combined pre+post)
for k = 1:3
    figure('Color','w'); hold on; grid on; box on
    mstep = max(1, floor(numel(t_full)/25));   % ~25 markers/curve
    h1 = plot(t_full, X_full1(k,:), 'Color', c1, 'LineWidth', 1.8, ...
        'LineStyle','-', 'Marker','o','MarkerSize',4, ...
        'MarkerIndices', 1:mstep:numel(t_full), ...
        'DisplayName','Susceptible (Set 1)');
    h2 = plot(t_full, X_full2(k,:), 'Color', c2, 'LineWidth', 1.8, ...
        'LineStyle','--','Marker','s','MarkerSize',4, ...
        'MarkerIndices', 1:mstep:numel(t_full), ...
        'DisplayName','Disease Carrying (Set 2)');
    hx = xline(t_perturb, '-k', 'DisplayName','Perturbation @ t=1800');
    xlabel('Time (months)'); ylabel(labels{k});
    title(sprintf('Species %s: Set 1 vs Set 2 (pre+post)', labels{k}));
    legend([h1 h2 hx], 'Location','best'); ylim([0 inf]);
end

% ====================== Lévy helper (vector state) ==========================
function x = levy_jump_update_vec(x, t, lambdaJ, aJ, dt, idx, tJ0, tJ1)
    % No jumps outside window or degenerate params
    if t < tJ0 || t > tJ1 || lambdaJ <= 0 || aJ <= 0 || isempty(idx)
        return;
    end
    % number of jump events this step
    N = poissrnd(lambdaJ * dt);
    if N == 0, return; end

    L = numel(idx);
    % Gaussian amplitudes; keep (1+J) > 0
    J = randn(L, N) * aJ;
    J = max(J, -0.9);

    % Small-jump log compensation: E[log(1+J)] ≈ -a^2/2
    logmult = sum(log1p(J), 2) + (N * (aJ^2)/2);
    mult    = exp(logmult);

    x(idx) = x(idx) .* mult;
end

