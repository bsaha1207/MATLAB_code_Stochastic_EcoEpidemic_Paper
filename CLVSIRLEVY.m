clc; clear; close all;

% ================================================================
% SDE CLV–SIR (two sets) with Lévy Jumps
%  - Pre: independent environments; Post: shared environment
%  - Brownian multiplicative noise + compound Poisson (Lévy) jumps
% ================================================================

% ========================= Time & Event =========================
t_perturb = 1500;
t_end     = 2500;
dt        = 0.002;

N_pre  = round((t_perturb - 0)/dt);
N_post = round((t_end - t_perturb)/dt);

t1     = linspace(0,         t_perturb, N_pre+1);
t2     = linspace(t_perturb, t_end,     N_post+1);
t_full = [t1, t2(2:end)];

% ========================= Base CLV Params ======================
% Period scaling
T_current = 2.85; T_target = 12; tau = T_current / T_target;

% Class-27 parameters
eta_c = 2/53 * (421 + sqrt(13471));
nu_c  = 64 - 3*eta_c;
eta   = eta_c + 1/20;
nu    = nu_c + 85/1e6;

A = tau * [48, 12, eta;
           60, 16, nu;
           84, 44, 32];
r = sum(A, 2);      % 3x1

% ========================= ICs (two sets) =======================
Y01 = [1.0800; 1.0500; 1.0100];
Y02 = [1.0300; 1.0000; 1.0200];

% SIR splits @ t=0 (Set 1: all susceptible; Set 2: small infection & env)
S10 = 1.00*Y01; I10 = 0.00*Y01; R10 = zeros(3,1); E10 = 0.0;
S20 = 0.99*Y02; I20 = 0.01*Y02; R20 = 0.00*Y02; E20 = 0.5;

x10 = [S10; I10; R10; E10];   % 10x1 Set 1
x20 = [S20; I20; R20; E20];   % 10x1 Set 2

% ========================= Epidemic Params ======================
Bscale = 0.001;
BETA1 = Bscale*A;    % within-set (pre)
BETA2 = BETA1;

alpha1 = [0.02; 0.015; 0.01];
alpha2 = alpha1;

gamma1 = [0.10; 0.11; 0.13];
gamma2 = gamma1;

mu1    = [0.01; 0.01; 0.01];
mu2    = [0.01; 0.01; 0.01];

sigma1 = [0.20; 0.20; 0.20];   % shedding to environment
sigma2 = sigma1;

delta1 = 0.30; 
delta2 = 0.30;

P1_pre = struct('A',A,'r',r,'BETA',BETA1,'alpha',alpha1,'gamma',gamma1,'mu',mu1,'sigma',sigma1,'delta',delta1);
P2_pre = struct('A',A,'r',r,'BETA',BETA2,'alpha',alpha2,'gamma',gamma2,'mu',mu2,'sigma',sigma2,'delta',delta2);

% ========================= SDE Noise Levels =====================
% --- Pre (visible test) ---
zS1_pre = 0.05; zI1_pre = 0.07; zR1_pre = 0.04; zE1_pre = 0.05;
zS2_pre = 0.05; zI2_pre = 0.07; zR2_pre = 0.04; zE2_pre = 0.05;

% --- Post (higher = extreme weather) ---
zS1_post = 2.0*zS1_pre; zI1_post = 2.0*zI1_pre; zR1_post = 2.0*zR1_pre;
zS2_post = 1.8*zS2_pre; zI2_post = 1.8*zI2_pre; zR2_post = 1.8*zR2_pre;
zE_post  = 2.0*max(zE1_pre, zE2_pre);

% Assemble multiplicative diffusion vectors (elementwise)
Zpre1 = [ zS1_pre*ones(3,1); zI1_pre*ones(3,1); zR1_pre*ones(3,1); zE1_pre ];  % 10x1
Zpre2 = [ zS2_pre*ones(3,1); zI2_pre*ones(3,1); zR2_pre*ones(3,1); zE2_pre ];  % 10x1
% Post mixed (19x1): [S1(3); I1(3); R1(3); E; S2(3); I2(3); R2(3)]
Zpost = [ zS1_post*ones(3,1); zI1_post*ones(3,1); zR1_post*ones(3,1); zE_post; ...
          zS2_post*ones(3,1); zI2_post*ones(3,1); zR2_post*ones(3,1) ];

% ========================= Lévy Jump Settings ====================
% Active windows (you can adjust)
tJ0_pre  = 1500;    tJ1_pre  = 1530;     % pre-event window
tJ0_post = 1650;  tJ1_post = 1680;   % post-event window

% Intensities (events per unit time) and amplitude std
lambdaJ_pre  = 0.20;  aJ_pre  = 0.20;   % "small" jump regime
lambdaJ_post = 0.60;  aJ_post = 0.80;   % "large" jump regime

% Target indices for multiplicative jumps
% Pre state: x = [S(3); I(3); R(3); E] -> size 10
jump_idx_pre  = [4:6, 10];                 % I(1:3) and E
% Post mixed: x = [S1(3); I1(3); R1(3); E; S2(3); I2(3); R2(3)] -> size 19
jump_idx_post = [4:6, 10, 14:16];          % I1(1:3), E, I2(1:3)

% ========================= PRE: SDE (Euler–Maruyama) ============
% x_{n+1} = x_n + f(x_n)dt + (Z .* x_n) .* dW  +  multiplicative Lévy jumps
rng(1);
dW_pre1 = sqrt(dt) * randn(10, N_pre);   % Set 1 (independent Brownian)
dW_pre2 = sqrt(dt) * randn(10, N_pre);   % Set 2

X1a = zeros(N_pre+1,10); X1a(1,:) = x10.';   % time x state
X2a = zeros(N_pre+1,10); X2a(1,:) = x20.';

for i = 1:N_pre
    % --- Set 1 ---
    x  = X1a(i,:).';                         % 10x1
    fx = sirclv_rhs(0, x, P1_pre);           % drift (10x1)
    x  = x + fx*dt + (Zpre1 .* x) .* dW_pre1(:,i);
    % Lévy jumps for Set 1
    t_now = t1(i);
    x  = levy_jump_update(x, t_now, lambdaJ_pre, aJ_pre, dt, jump_idx_pre, tJ0_pre, tJ1_pre);
    X1a(i+1,:) = max(x,0).';

    % --- Set 2 ---
    y  = X2a(i,:).';
    fy = sirclv_rhs(0, y, P2_pre);
    y  = y + fy*dt + (Zpre2 .* y) .* dW_pre2(:,i);
    % Lévy jumps for Set 2
    y  = levy_jump_update(y, t_now, lambdaJ_pre, aJ_pre, dt, jump_idx_pre, tJ0_pre, tJ1_pre);
    X2a(i+1,:) = max(y,0).';
end

% Values at t = 1500-
S1a_end = X1a(end,1:3).'; I1a_end = X1a(end,4:6).'; R1a_end = X1a(end,7:9).'; E1a_end = X1a(end,10);
S2a_end = X2a(end,1:3).'; I2a_end = X2a(end,4:6).'; R2a_end = X2a(end,7:9).'; E2a_end = X2a(end,10);

% ========================= EVENT: A,r perturb ===================
A_post = A; r_post = r;
A_post(1,1) = A(1,1) - 0.08*r(2);
A_post(2,1) = A(2,1) + 0.08*r(1);
r_post(1)   = r(1)   - 0.08*A(2,1);
r_post(2)   = r(2)   - 0.08*A(1,1);

% ========================= POST Params (mixed) ==================
scale_within_beta = 1.8;   % within-set β
cross_frac        = 0.6;   % cross-set fraction
alpha_scale       = 1.5;   % E -> host
sigma_scale       = 1.3;   % shedding
delta_shared      = 0.15;  % env decay
gamma_scale       = 0.85;  % recovery
mu_scale          = 1.25;  % mortality

B11 = scale_within_beta * BETA1;       % Set1-within
B22 = scale_within_beta * BETA2;       % Set2-within
B12 = cross_frac * B11;                % Set2 -> Set1
B21 = cross_frac * B22;                % Set1 -> Set2

alpha1_post = alpha_scale * alpha1;  alpha2_post = alpha_scale * alpha2;
sigma1_post = sigma_scale * sigma1;  sigma2_post = sigma_scale * sigma2;
gamma1_post = gamma_scale * gamma1;  gamma2_post = gamma_scale * gamma2;
mu1_post    = mu_scale    * mu1;     mu2_post    = mu_scale    * mu2;

E_shared0 = E1a_end + E2a_end;
x_post0   = [S1a_end; I1a_end; R1a_end; E_shared0; S2a_end; I2a_end; R2a_end];

% ========================= POST: SDE (EM, mixed) ================
rng(2);
dW_post = sqrt(dt) * randn(19, N_post);

Xb = zeros(N_post+1, 19);   Xb(1,:) = x_post0.';

for k = 1:N_post
    x  = Xb(k,:).';   % 19x1
    fx = sirclv_rhs_mixed(0, x, ...
         A_post, r_post, B11, B12, B21, B22, ...
         alpha1_post, alpha2_post, gamma1_post, gamma2_post, ...
         mu1_post, mu2_post, sigma1_post, sigma2_post, delta_shared);
    x  = x + fx*dt + (Zpost .* x) .* dW_post(:,k);
    % Lévy jumps for mixed system
    t_now = t2(k);
    x  = levy_jump_update(x, t_now, lambdaJ_post, aJ_post, dt, jump_idx_post, tJ0_post, tJ1_post);
    Xb(k+1,:) = max(x,0).';
end

% ========================= Unpack + Merge =======================
% Pre (time x state)
S1a = X1a(:,1:3); I1a = X1a(:,4:6); R1a = X1a(:,7:9); E1a = X1a(:,10);
S2a = X2a(:,1:3); I2a = X2a(:,4:6); R2a = X2a(:,7:9); E2a = X2a(:,10);

% Post (time x state)
S1b = Xb(:,1:3);  I1b = Xb(:,4:6);  R1b = Xb(:,7:9);  Eb  = Xb(:,10);
S2b = Xb(:,11:13); I2b = Xb(:,14:16); R2b = Xb(:,17:19);

% Merge (drop first post row to avoid duplicate time)
S1f = [S1a; S1b(2:end,:)]; I1f = [I1a; I1b(2:end,:)]; R1f = [R1a; R1b(2:end,:)];
S2f = [S2a; S2b(2:end,:)]; I2f = [I2a; I2b(2:end,:)]; R2f = [R2a; R2b(2:end,:)];
E1f = [E1a; Eb(2:end,:)];  E2f = [E2a; Eb(2:end,:)];

% ========================= Plotting =============================
% Prevalence helper (safe denominator)
prev = @(I,S,R) 100*I ./ max(I+S+R, eps);

labels = {'Species 1','Species 2','Species 3'};

% (A) Prevalence: one figure per species (Set 1 vs Set 2)
for k = 1:3
    figure('Color','w'); hold on; grid on; box on
    h1 = plot(t_full, prev(I1f(:,k), S1f(:,k), R1f(:,k)), '-',  ...
              'Color',[0.64,0.08,0.18], 'LineWidth',1.8, 'DisplayName','Set 1');
    h2 = plot(t_full, prev(I2f(:,k), S2f(:,k), R2f(:,k)), '--', ...
              'Color',[0.64,0.08,0.18], 'LineWidth',1.8, 'DisplayName','Set 2');
    he = xline(t_perturb,'-.','Color',[0.4 0.4 0.4], 'LineWidth',2.0, 'DisplayName','Perturbation');
    xlabel('Time (Months)'); ylabel('Prevalence (%)');
    title([labels{k} ' — Prevalence (Set 1 vs Set 2)']);
    legend([h1 h2 he], 'Location','best'); ylim([0 inf]);
end

% (B) S, I, R: one figure per species (both sets)
for k = 1:3
    figure('Color','w'); hold on; grid on; box on
    % S
    s1 = plot(t_full, S1f(:,k), '-',  'Color',[0 0.4470 0.7410], 'LineWidth',1.8, 'DisplayName','Set 1: S');
    s2 = plot(t_full, S2f(:,k), '--', 'Color',[0 0.4470 0.7410], 'LineWidth',1.8, 'DisplayName','Set 2: S');
    % I
    i1 = plot(t_full, I1f(:,k), '-',  'Color',[0.64,0.08,0.18],  'LineWidth',1.8, 'DisplayName','Set 1: I');
    i2 = plot(t_full, I2f(:,k), '--', 'Color',[0.64,0.08,0.18],  'LineWidth',1.8, 'DisplayName','Set 2: I');
    % R
    r1p = plot(t_full, R1f(:,k), '-',  'Color',[0.85,0.33,0.10], 'LineWidth',1.8, 'DisplayName','Set 1: R');
    r2p = plot(t_full, R2f(:,k), '--', 'Color',[0.85,0.33,0.10], 'LineWidth',1.8, 'DisplayName','Set 2: R');
    he = xline(t_perturb,'-.','Color',[0.4 0.4 0.4], 'LineWidth',2.0, 'DisplayName','Perturbation');
    xlabel('Time (Months)'); ylabel('Population');
    title([labels{k} ' — S, I, R (both sets)']);
    legend([s1 s2 i1 i2 r1p r2p he], 'Location','best'); ylim([0 inf]);
end

% (C) Environment (optional context)
figure('Color','w'); hold on; grid on; box on
plot(t_full, E1f, '-',  'LineWidth',1.6, 'DisplayName','Env (ref 1)');
plot(t_full, E2f, '--', 'LineWidth',1.6, 'DisplayName','Env (ref 2)');
xline(t_perturb,'-.','Color',[0.4 0.4 0.4],'LineWidth',2.0);
xlabel('Time'); ylabel('Environment'); legend('Location','best'); ylim([0 inf]);


% ---- Post-perturbation window ----
t0 = 1500; t1 = 2500;
idx = (t_full >= t0) & (t_full <= t1);

% Prevalence per species (columns = species 1,2,3)
P1 = [ prev(I1f(idx,1), S1f(idx,1), R1f(idx,1)), ...
       prev(I1f(idx,2), S1f(idx,2), R1f(idx,2)), ...
       prev(I1f(idx,3), S1f(idx,3), R1f(idx,3)) ];

P2 = [ prev(I2f(idx,1), S2f(idx,1), R2f(idx,1)), ...
       prev(I2f(idx,2), S2f(idx,2), R2f(idx,2)), ...
       prev(I2f(idx,3), S2f(idx,3), R2f(idx,3)) ];

% ---- 3D phase portrait: (species1%, species2%, species3%) ----
figure('Color','w'); hold on; grid on; box on

% Trajectories
h1 = plot3(P1(:,1), P1(:,2), P1(:,3), '-',  'LineWidth', 2.0, 'Color', [0.05 0.35 0.75]);
h2 = plot3(P2(:,1), P2(:,2), P2(:,3), '--', 'LineWidth', 2.0, 'Color', [0.75 0.20 0.10]);

% Mark start/end points
%plot3(P1(1,1), P1(1,2), P1(1,3), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [0.05 0.35 0.75], 'MarkerEdgeColor','k');
%plot3(P1(end,1), P1(end,2), P1(end,3), 's', 'MarkerSize', 7, 'MarkerFaceColor', [0.05 0.35 0.75], 'MarkerEdgeColor','k');

% Axes/labels/limits
xlabel('Prevalence: Species 1 (%)');
ylabel('Prevalence: Species 2 (%)');
zlabel('Prevalence: Species 3 (%)');
title(sprintf('Post-perturbation prevalence phase portrait (t \\in [%d,%d])', t0, t1));
%xlim([0 100]); ylim([0 100]); zlim([0 100]);
%view(3); 
view(135,25);% adjust viewing angle for clarity
legend([h1 h2], {'N.D.C. Set','D.C. Set'}, 'Location','best');
% ========================= RHS (drift) ==========================
% Pre-event RHS: CLV-SIR with independent environment (within-set only)
% State x = [S(3); I(3); R(3); E]
function dx = sirclv_rhs(~, x, P)
    S = x(1:3); I = x(4:6); R = x(7:9); E = x(10);
    r = P.r(:); a = P.alpha(:); g = P.gamma(:); m = P.mu(:); s = P.sigma(:);

    Y = S + I + R;                 % 3x1
    G = r - P.A * Y;               % 3x1
    lambda = P.BETA*I + a*E;       % 3x1

    dS = S.*G - S.*lambda;
    dI = I.*G + S.*lambda - (g+m).*I;
    dR = R.*G + g.*I;
    dE = dot(s, I) - P.delta*E;

    dx = [dS; dI; dR; dE];
end

% Post-event mixed RHS: shared environment, cross-set transmission
% State x (19x1) = [S1(3); I1(3); R1(3); E; S2(3); I2(3); R2(3)]
function dx = sirclv_rhs_mixed(~, x, ...
    A_post, r_post, B11, B12, B21, B22, ...
    alpha1, alpha2, gamma1, gamma2, mu1, mu2, ...
    sigma1, sigma2, delta_shared)

    % Unpack
    S1 = x(1:3);  I1 = x(4:6);  R1 = x(7:9);
    E  = x(10);
    S2 = x(11:13); I2 = x(14:16); R2 = x(17:19);

    % Columnize
    r  = r_post(:);
    a1 = alpha1(:); a2 = alpha2(:);
    g1 = gamma1(:); g2 = gamma2(:);
    m1 = mu1(:);    m2 = mu2(:);
    s1 = sigma1(:); s2 = sigma2(:);

    % CLV growth (within each set)
    Y1 = S1 + I1 + R1;  G1 = r - A_post*Y1;
    Y2 = S2 + I2 + R2;  G2 = r - A_post*Y2;

    % Forces of infection (cross-set + shared E)
    lambda1 = B11*I1 + B12*I2 + a1*E;
    lambda2 = B21*I1 + B22*I2 + a2*E;

    % SIR
    dS1 = S1.*G1 - S1.*lambda1;
    dI1 = I1.*G1 + S1.*lambda1 - (g1+m1).*I1;
    dR1 = R1.*G1 + g1.*I1;

    dS2 = S2.*G2 - S2.*lambda2;
    dI2 = I2.*G2 + S2.*lambda2 - (g2+m2).*I2;
    dR2 = R2.*G2 + g2.*I2;

    % Shared environment
    dE = dot(s1,I1) + dot(s2,I2) - delta_shared*E;

    dx = [dS1; dI1; dR1; dE; dS2; dI2; dR2];
end

% ========================= Lévy Jump Helper ======================
% Multiplicative compound-Poisson jumps on selected indices.
% x(idx) <- x(idx) .* prod_{events}(1 + J), J ~ N(0, aJ^2), truncated > -0.9
function x = levy_jump_update(x, t, lambdaJ, aJ, dt, idx, tJ0, tJ1)
    if t < tJ0 || t > tJ1 || lambdaJ <= 0 || aJ <= 0 || isempty(idx)
        return;
    end
    % number of jump events this step
    N = poissrnd(lambdaJ * dt);
    if N == 0, return; end

    % draw jump amplitudes for each targeted component and each event
    L = numel(idx);
    J = randn(L, N) * aJ;                   % mean 0 (compensated)
    J = max(J, -0.9);                        % keep (1+J) > 0

    % multiplicative effect per component across N events
    mult = prod(1 + J, 2);% Lx1
    %logmult = sum(log1p(J), 2) + (N * (aJ^2)/2);
    %mult    = exp(logmult);
    x(idx) = x(idx) .* mult;                 % apply multiplicatively
end
%%

% ---- Species totals per set (time x species) ----
Y1f = S1f + I1f + R1f;   % Set 1 totals
Y2f = S2f + I2f + R2f;   % Set 2 totals

labels = {'Species 1','Species 2','Species 3'};
c1 = [0 0.4470 0.7410];
c2 = [0.4940 0.1840 0.5560];
figure('Color','w'); hold on; grid on; box on
step = 10;   % decimate for readability
plot3(Y1f(1:step:end,1), Y1f(1:step:end,2), Y1f(1:step:end,3), '-',  ...
      'Color',c1,'LineWidth',1.6,'DisplayName','Set 1 (total)');
plot3(Y2f(1:step:end,1), Y2f(1:step:end,2), Y2f(1:step:end,3), '--', ...
      'Color',c2,'LineWidth',1.6,'DisplayName','Set 2 (total)');
xlabel('Y_1'); ylabel('Y_2'); zlabel('Y_3');
title('Species totals: Stochastic Phase Portrait (Baseline \rightarrow Perturbation)');
legend('Location','best'); view(135,25); axis tight
for k = 1:3
    figure('Color','w'); hold on; grid on; box on
    mstep = max(1, floor(numel(t_full)/25));
    h1 = plot(t_full, Y1f(:,k), 'Color', c1, 'LineWidth', 1.8, ...
        'LineStyle','-', 'Marker','o', 'MarkerSize', 4, ...
        'MarkerIndices', 1:mstep:numel(t_full), ...
        'DisplayName','Set 1 (total)');
    h2 = plot(t_full, Y2f(:,k), 'Color', c2, 'LineWidth', 1.8, ...
        'LineStyle','--','Marker','s', 'MarkerSize', 4, ...
        'MarkerIndices', 1:mstep:numel(t_full), ...
        'DisplayName','Set 2 (total)');
    hx = xline(t_perturb, '-k', 'DisplayName', sprintf('Perturbation @ t=%g', t_perturb));
    xlabel('Time (months)'); ylabel(labels{k});
    title(sprintf('%s: Set 1 vs Set 2 (species totals)', labels{k}));
    legend([h1 h2 hx], 'Location','best'); ylim([0 inf]);
end
idx_post = t_full >= t_perturb;           % post-event window
step     = 10;                            % decimate for clarity

Y1p = Y1f(idx_post, :);
Y2p = Y2f(idx_post, :);

figure('Color','w'); hold on; grid on; box on

p1 = plot3(Y1p(1:step:end,1), Y1p(1:step:end,2), Y1p(1:step:end,3), ...
    '-',  'Color', c1, 'LineWidth', 1.8, 'DisplayName', 'Set 1 (post)');
p2 = plot3(Y2p(1:step:end,1), Y2p(1:step:end,2), Y2p(1:step:end,3), ...
    '--', 'Color', c2, 'LineWidth', 1.8, 'DisplayName', 'Set 2 (post)');
xlabel('Y_1'); ylabel('Y_2'); zlabel('Y_3');
title(sprintf('Species Totals: Post-event'));
legend('Location','best');
view(135,25); 


