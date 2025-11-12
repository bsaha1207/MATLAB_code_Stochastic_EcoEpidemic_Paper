clc; clear; close all;

% ================================================================
% Spatial SDE CLV–SIR (two sets) with Lévy Jumps (1D)
% - Space: 1D domain with diffusion (D) and advection (V)
% - Time: 0 --> t_end with a perturbation at t_perturb
% - Pre: independent environments; Post: shared environment
% - Integrator: Method-of-lines + Euler–Maruyama
% - BCs: Zero-flux (Neumann); advection with upwind
% ================================================================

% ========================= Time & Event =========================
t_perturb = 1500;
t_end     = 2500;
dt        = 0.01;             % (larger than your SDE-only; lower dt => more stable)
rng(1);                       % global seed for reproducibility

N_pre  = round((t_perturb - 0)/dt);
N_post = round((t_end - t_perturb)/dt);

t1      = linspace(0,          t_perturb,  N_pre+1);
t2      = linspace(t_perturb,  t_end,      N_post+1);
t_full  = [t1, t2(2:end)];
Nt_full = numel(t_full);

% ========================= Space Grid ===========================
Lx = 100;              % domain length
nx = 61;               % spatial nodes (>= 41 recommended)
x  = linspace(0, Lx, nx).';
dx = x(2) - x(1);

% Preallocate upwind helpers once
[E_fw, E_bw] = deal(speye(nx)); % (identities; used in directional upwind helper)

% Helper handles
lap = @(U) neumann_laplacian(U, dx);     % second derivative with Neumann BC
d1  = @(U,V) upwind_first_derivative(U, V, dx); % upwind first derivative (vector V allowed)

% ========================= Base CLV Params ======================
T_current = 2.85;
T_target  = 12;
tau       = T_current / T_target;
eta_c     = 2/53 * (421 + sqrt(13471));
nu_c      = 64 - 3*eta_c;
eta       = eta_c + 1/20;
nu        = nu_c + 85/1e6;

A = tau * [48, 12, eta;
           60, 16, nu;
           84, 44, 32];
r = sum(A, 2); % 3x1

% ========================= ICs (two sets) =======================
Y01 = [1.0800; 1.0500; 1.0100];
Y02 = [1.0300; 1.0000; 1.0200];

% Spread ICs across space (slight spatial inhomogeneity)
bump = 1 + 0.02*cos(2*pi*x/Lx);

S10 = Y01 .* bump';
I10 = 0*Y01 .* bump';
R10 = zeros(3, nx);

S20 = 0.90*Y02 .* bump';
I20 = 0.10*Y02 .* bump';
R20 = 0.00*Y02 .* bump';

E10 = 0.0*ones(1, nx);
E20 = 0.5*exp(-((x-Lx/2)/(Lx/8)).^2)'; % localized env for set 2 initially

% ========================= Epidemic Params ======================
Bscale = 0.003;
BETA1  = Bscale*A; % within-set (pre)
BETA2  = BETA1;

alpha1 = [0.02; 0.015; 0.01];
alpha2 = alpha1;

gamma1 = [0.10; 0.11; 0.13];
gamma2 = gamma1;

mu1 = [0.01; 0.01; 0.01];
mu2 = [0.01; 0.01; 0.01];

sigma1 = [0.20; 0.20; 0.20]; % shedding to environment
sigma2 = sigma1;

delta1 = 0.30;
delta2 = 0.30;

P1_pre = struct('A',A,'r',r,'BETA',BETA1,'alpha',alpha1,'gamma',gamma1,'mu',mu1,'sigma',sigma1,'delta',delta1);
P2_pre = struct('A',A,'r',r,'BETA',BETA2,'alpha',alpha2,'gamma',gamma2,'mu',mu2,'sigma',sigma2,'delta',delta2);

% ========================= Spatial Transport ====================
% Species-specific diffusion & advection (choose your baselines)
D_S = [0.04; 0.02; 0.01]; % S diffusion for species 1..3
D_I = [0.005; 0.005; 0.002]; % I diffusion
D_R = [0.03; 0.02; 0.01]; % R diffusion
D_E = 0.02; % Environment diffusion (scalar)

V_S = [0.10; 0.08; 0.05]; % S advection speeds (>0 => flow to +x)
V_I = [0.006; 0.005; 0.004]; % I advection
V_R = [0.04; 0.03; 0.02]; % R advection
V_E = 0.01; % Environment advection

% ========================= SDE Noise Levels =====================
% --- Pre (visible test) ---
zS1_pre = 0.05; zI1_pre = 0.07; zR1_pre = 0.04; zE1_pre = 0.05;
zS2_pre = 0.05; zI2_pre = 0.07; zR2_pre = 0.04; zE2_pre = 0.05;

% --- Post (higher = extreme weather) ---
zS1_post = 2.0*zS1_pre; zI1_post = 2.0*zI1_pre; zR1_post = 2.0*zR1_pre;
zS2_post = 1.8*zS2_pre; zI2_post = 1.8*zI2_pre; zR2_post = 1.8*zR2_pre;
zE_post  = 2.0*max(zE1_pre, zE2_pre);

% ========================= Lévy Jump Settings ====================
tJ0_pre = 10;   tJ1_pre = 30;    % pre-event window
tJ0_post = 1600; tJ1_post = 2000; % post-event window
lambdaJ_pre = 0.20; aJ_pre = 0.20; % "small" jump regime
lambdaJ_post = 0.60; aJ_post = 0.80; % "large" jump regime
% lambdaJ_pre = 0.00; aJ_pre = 0.00;
% lambdaJ_post = 0.00; aJ_post = 0.00;

% Jumped components: I(1:3) and E for each set, at all x nodes
jump_targets_pre  = struct('I',  true(3,1), 'E', true);
jump_targets_post = struct('I1', true(3,1), 'I2', true(3,1), 'E', true);

% ========================= Storage (Pre) ========================
S1a = zeros(N_pre+1, nx, 3); I1a = S1a; R1a = S1a; E1a = zeros(N_pre+1, nx);
S2a = S1a;                   I2a = S1a; R2a = S1a; E2a = zeros(N_pre+1, nx);

S1a(1,:,:) = S10.'; I1a(1,:,:) = I10.'; R1a(1,:,:) = R10.'; E1a(1,:) = E10;
S2a(1,:,:) = S20.'; I2a(1,:,:) = I20.'; R2a(1,:,:) = R20.'; E2a(1,:) = E20.';

% ========================= PRE: Integrate =======================
for n = 1:N_pre
    t = t1(n);

    % --- Set 1 ---
    S = squeeze(S1a(n,:,:)).'; % 3 x nx
    I = squeeze(I1a(n,:,:)).';
    R = squeeze(R1a(n,:,:)).';
    E = E1a(n,:);             % 1 x nx

    [dS, dI, dR, dE] = rhs_local(S, I, R, E, P1_pre); % local epi drift

    % Add spatial transport (componentwise by species)
    for sp = 1:3
        dS(sp,:) = dS(sp,:) + D_S(sp)*lap(S(sp,:)) - d1(S(sp,:), V_S(sp));
        dI(sp,:) = dI(sp,:) + D_I(sp)*lap(I(sp,:)) - d1(I(sp,:), V_I(sp));
        dR(sp,:) = dR(sp,:) + D_R(sp)*lap(R(sp,:)) - d1(R(sp,:), V_R(sp));
    end
    dE = dE + D_E*lap(E) - d1(E, V_E);

    % Multiplicative Brownian noise (nodewise)
    dW_S = sqrt(dt)*randn(3, nx)*zS1_pre;
    dW_I = sqrt(dt)*randn(3, nx)*zI1_pre;
    dW_R = sqrt(dt)*randn(3, nx)*zR1_pre;
    dW_E = sqrt(dt)*randn(1, nx)*zE1_pre;

    S = S + dS*dt + (dW_S .* S);
    I = I + dI*dt + (dW_I .* I);
    R = R + dR*dt + (dW_R .* R);
    E = E + dE*dt + (dW_E .* E);

    % Lévy jumps (multiplicative) during pre window
    [I, E] = levy_jumps_space(I, E, t, lambdaJ_pre, aJ_pre, dt, jump_targets_pre);

    % Positivity
    S = max(S,0); I = max(I,0); R = max(R,0); E = max(E,0);

    % Save
    S1a(n+1,:,:) = S.'; I1a(n+1,:,:) = I.'; R1a(n+1,:,:) = R.'; E1a(n+1,:) = E;

    % --- Set 2 ---
    S = squeeze(S2a(n,:,:)).'; % 3 x nx
    I = squeeze(I2a(n,:,:)).';
    R = squeeze(R2a(n,:,:)).';
    E = E2a(n,:);

    [dS, dI, dR, dE] = rhs_local(S, I, R, E, P2_pre);

    for sp = 1:3
        dS(sp,:) = dS(sp,:) + D_S(sp)*lap(S(sp,:)) - d1(S(sp,:), V_S(sp));
        dI(sp,:) = dI(sp,:) + D_I(sp)*lap(I(sp,:)) - d1(I(sp,:), V_I(sp));
        dR(sp,:) = dR(sp,:) + D_R(sp)*lap(R(sp,:)) - d1(R(sp,:), V_R(sp));
    end
    dE = dE + D_E*lap(E) - d1(E, V_E);

    dW_S = sqrt(dt)*randn(3, nx)*zS2_pre;
    dW_I = sqrt(dt)*randn(3, nx)*zI2_pre;
    dW_R = sqrt(dt)*randn(3, nx)*zR2_pre;
    dW_E = sqrt(dt)*randn(1, nx)*zE2_pre;

    S = S + dS*dt + (dW_S .* S);
    I = I + dI*dt + (dW_I .* I);
    R = R + dR*dt + (dW_R .* R);
    E = E + dE*dt + (dW_E .* E);

    [I, E] = levy_jumps_space(I, E, t, lambdaJ_pre, aJ_pre, dt, jump_targets_pre);

    S = max(S,0); I = max(I,0); R = max(R,0); E = max(E,0);

    S2a(n+1,:,:) = S.'; I2a(n+1,:,:) = I.'; R2a(n+1,:,:) = R.'; E2a(n+1,:) = E;
end

% Values at t = 1500-
S1_end = squeeze(S1a(end,:,:)).';
I1_end = squeeze(I1a(end,:,:)).';
R1_end = squeeze(R1a(end,:,:)).';
E1_end = E1a(end,:);

S2_end = squeeze(S2a(end,:,:)).';
I2_end = squeeze(I2a(end,:,:)).';
R2_end = squeeze(R2a(end,:,:)).';
E2_end = E2a(end,:);

% ========================= EVENT: A,r perturb ===================
A_post = A; r_post = r;
A_post(1,1) = A(1,1) - 0.08*r(2);
A_post(2,1) = A(2,1) + 0.08*r(1);
r_post(1)   = r(1) - 0.08*A(2,1);
r_post(2)   = r(2) - 0.08*A(1,1);

% ========================= POST Params (mixed) ==================
scale_within_beta = 1.8;  % within-set β
cross_frac        = 0.6;  % cross-set fraction
alpha_scale       = 1.5;  % E -> host
sigma_scale       = 1.3;  % shedding
delta_shared      = 0.15; % env decay
gamma_scale       = 0.85; % recovery
mu_scale          = 1.25; % mortality

B11 = scale_within_beta * BETA1; % Set1-within
B22 = scale_within_beta * BETA2; % Set2-within
B12 = cross_frac * B11;          % Set2 -> Set1
B21 = cross_frac * B22;          % Set1 -> Set2

alpha1_post = alpha_scale * alpha1;
alpha2_post = alpha_scale * alpha2;
sigma1_post = sigma_scale * sigma1;
sigma2_post = sigma_scale * sigma2;
gamma1_post = gamma_scale * gamma1;
gamma2_post = gamma_scale * gamma2;
mu1_post    = mu_scale    * mu1;
mu2_post    = mu_scale    * mu2;

% Shared environment initial field
E_shared0 = E1_end + E2_end;

% ========================= Storage (Post) =======================
S1b = zeros(N_post+1, nx, 3); I1b = S1b; R1b = S1b;
S2b = S1b;                    I2b = S1b; R2b = S1b;
Eb  = zeros(N_post+1, nx);

S1b(1,:,:) = S1_end.'; I1b(1,:,:) = I1_end.'; R1b(1,:,:) = R1_end.';
S2b(1,:,:) = S2_end.'; I2b(1,:,:) = I2_end.'; R2b(1,:,:) = R2_end.';
Eb(1,:)    = E_shared0;

% ========================= POST: Integrate (mixed) ==============
for n = 1:N_post
    t = t2(n);

    S1 = squeeze(S1b(n,:,:)).';
    I1 = squeeze(I1b(n,:,:)).';
    R1 = squeeze(R1b(n,:,:)).';

    S2 = squeeze(S2b(n,:,:)).';
    I2 = squeeze(I2b(n,:,:)).';
    R2 = squeeze(R2b(n,:,:)).';

    E  = Eb(n,:);

    % Local epi drift (mixed, per x-node)
    [dS1, dI1, dR1, dS2, dI2, dR2, dE] = rhs_mixed_local( ...
        S1,I1,R1,S2,I2,R2,E, ...
        A_post, r_post, ...
        B11,B12,B21,B22, ...
        alpha1_post,alpha2_post, ...
        gamma1_post,gamma2_post, ...
        mu1_post,mu2_post, ...
        sigma1_post,sigma2_post, ...
        delta_shared);

    % Spatial transport
    for sp = 1:3
        dS1(sp,:) = dS1(sp,:) + D_S(sp)*lap(S1(sp,:)) - d1(S1(sp,:), V_S(sp));
        dI1(sp,:) = dI1(sp,:) + D_I(sp)*lap(I1(sp,:)) - d1(I1(sp,:), V_I(sp));
        dR1(sp,:) = dR1(sp,:) + D_R(sp)*lap(R1(sp,:)) - d1(R1(sp,:), V_R(sp));

        dS2(sp,:) = dS2(sp,:) + D_S(sp)*lap(S2(sp,:)) - d1(S2(sp,:), V_S(sp));
        dI2(sp,:) = dI2(sp,:) + D_I(sp)*lap(I2(sp,:)) - d1(I2(sp,:), V_I(sp));
        dR2(sp,:) = dR2(sp,:) + D_R(sp)*lap(R2(sp,:)) - d1(R2(sp,:), V_R(sp));
    end
    dE = dE + D_E*lap(E) - d1(E, V_E);

    % Noise (post, higher)
    dW_S1 = sqrt(dt)*randn(3, nx)*zS1_post;
    dW_I1 = sqrt(dt)*randn(3, nx)*zI1_post;
    dW_R1 = sqrt(dt)*randn(3, nx)*zR1_post;

    dW_S2 = sqrt(dt)*randn(3, nx)*zS2_post;
    dW_I2 = sqrt(dt)*randn(3, nx)*zI2_post;
    dW_R2 = sqrt(dt)*randn(3, nx)*zR2_post;

    dW_E  = sqrt(dt)*randn(1, nx)*zE_post;

    % Euler–Maruyama update
    S1 = S1 + dS1*dt + (dW_S1 .* S1);
    I1 = I1 + dI1*dt + (dW_I1 .* I1);
    R1 = R1 + dR1*dt + (dW_R1 .* R1);

    S2 = S2 + dS2*dt + (dW_S2 .* S2);
    I2 = I2 + dI2*dt + (dW_I2 .* I2);
    R2 = R2 + dR2*dt + (dW_R2 .* R2);

    E  = E  + dE*dt  + (dW_E  .* E );

    % Lévy jumps (post window) applied at all nodes for I1, I2, and E
    [I1, I2, E] = levy_jumps_space_mixed(I1,I2,E, t, lambdaJ_post, aJ_post, dt, jump_targets_post);

    % Positivity
    S1 = max(S1,0); I1 = max(I1,0); R1 = max(R1,0);
    S2 = max(S2,0); I2 = max(I2,0); R2 = max(R2,0); E = max(E ,0);

    % Save
    S1b(n+1,:,:) = S1.'; I1b(n+1,:,:) = I1.'; R1b(n+1,:,:) = R1.';
    S2b(n+1,:,:) = S2.'; I2b(n+1,:,:) = I2.'; R2b(n+1,:,:) = R2.';
    Eb(n+1,:)    = E;
end

% ========================= Merge Full Fields ====================
% (We just need I,S,R & E to make prevalence surfaces)
S1f = [S1a; S1b(2:end,:,:)];  I1f = [I1a; I1b(2:end,:,:)];  R1f = [R1a; R1b(2:end,:,:)];
S2f = [S2a; S2b(2:end,:,:)];  I2f = [I2a; I2b(2:end,:,:)];  R2f = [R2a; R2b(2:end,:,:)];
E1f = [E1a; Eb(2:end,:)];     % for plotting: set 1 environment
E2f = [E2a; Eb(2:end,:)];     % for plotting: set 2 environment (post=shared)

% ========================= Surf Plots ===========================
% Prevalence (%) = 100 * I ./ (S+I+R)
prev = @(I,S,R) 100*(I ./ (S+I+R));
[Xgrid,Tgrid] = meshgrid(x, t_full);
labels = {'Species 1','Species 2','Species 3'};

% Six prevalence surfaces: species 1..3 for each set
for sp = 1:3
    % Set 1
    P = prev(squeeze(I1f(:,:,sp)), squeeze(S1f(:,:,sp)), squeeze(R1f(:,:,sp)));
    figure('Color','w'); surf(Xgrid, Tgrid, P, 'EdgeColor','none'); view(3); colormap(parula); colorbar;
    xlabel('Space (x)'); ylabel('Time'); title(sprintf('%s — Prevalence (Set 1) [%%]', labels{sp}));
    hold off;

    % Set 2
    P = prev(squeeze(I2f(:,:,sp)), squeeze(S2f(:,:,sp)), squeeze(R2f(:,:,sp)));
    figure('Color','w'); surf(Xgrid, Tgrid, P, 'EdgeColor','none'); view(3); colormap(parula); colorbar;
    xlabel('Space (x)'); ylabel('Time'); title(sprintf('%s — Prevalence (Set 2) [%%]', labels{sp}));
    hold off;
end

% Two environment surfaces (post both show shared E)
figure('Color','w');
surf(Xgrid, Tgrid, E1f, 'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); title('Environment — Set 1 reference');
hold on; yline(t_perturb,'k--','LineWidth',1.5); hold off;

figure('Color','w');
surf(Xgrid, Tgrid, E2f, 'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); title('Environment — Set 2 reference');
hold on; yline(t_perturb,'k--','LineWidth',1.5); hold off;

% ========================= Helper Functions =====================
function [dS,dI,dR,dE] = rhs_local(S,I,R,E,P)
% Local (nodewise) CLV–SIR (within each set) w/ independent environment
% Inputs: S,I,R are 3×nx, E is 1×nx
nx = size(S,2);
dS = zeros(3,nx); dI = dS; dR = dS; dE = zeros(1,nx);
for j = 1:nx
    Sj = S(:,j); Ij = I(:,j); Rj = R(:,j); Ej = E(1,j);
    Y = Sj + Ij + Rj;                % 3x1
    G = P.r(:) - P.A * Y;            % 3x1
    lambda = P.BETA*Ij + P.alpha(:)*Ej; % 3x1
    dS(:,j) = Sj.*G - Sj.*lambda;
    dI(:,j) = Ij.*G + Sj.*lambda - (P.gamma(:)+P.mu(:)).*Ij;
    dR(:,j) = Rj.*G + P.gamma(:).*Ij;
    dE(1,j) = dot(P.sigma(:), Ij) - P.delta*Ej;
end
end

function [dS1,dI1,dR1,dS2,dI2,dR2,dE] = rhs_mixed_local( ...
    S1,I1,R1,S2,I2,R2,E, ...
    A, r, B11,B12,B21,B22, a1,a2, g1,g2, m1,m2, s1,s2, delta_shared)
% Mixed (post) local drift with shared environment and cross-set transmission
nx = size(S1,2);
dS1 = zeros(3,nx); dI1 = dS1; dR1 = dS1;
dS2 = zeros(3,nx); dI2 = dS2; dR2 = dS2;
dE  = zeros(1,nx);
for j = 1:nx
    S1j = S1(:,j); I1j = I1(:,j); R1j = R1(:,j);
    S2j = S2(:,j); I2j = I2(:,j); R2j = R2(:,j);
    Ej  = E(1,j);

    Y1 = S1j + I1j + R1j;  G1 = r(:) - A*Y1;
    Y2 = S2j + I2j + R2j;  G2 = r(:) - A*Y2;

    lambda1 = B11*I1j + B12*I2j + a1(:)*Ej;
    lambda2 = B21*I1j + B22*I2j + a2(:)*Ej;

    dS1(:,j) = S1j.*G1 - S1j.*lambda1;
    dI1(:,j) = I1j.*G1 + S1j.*lambda1 - (g1(:)+m1(:)).*I1j;
    dR1(:,j) = R1j.*G1 + g1(:).*I1j;

    dS2(:,j) = S2j.*G2 - S2j.*lambda2;
    dI2(:,j) = I2j.*G2 + S2j.*lambda2 - (g2(:)+m2(:)).*I2j;
    dR2(:,j) = R2j.*G2 + g2(:).*I2j;

    dE(1,j) = dot(s1(:),I1j) + dot(s2(:),I2j) - delta_shared*Ej;
end
end

function Lu = neumann_laplacian(u, dx)
% Second derivative with zero-flux (Neumann) BC using central differences
% u: 1×nx
nx = numel(u);
Lu = zeros(1,nx);
% interior
Lu(2:nx-1) = (u(3:nx) - 2*u(2:nx-1) + u(1:nx-2)) / dx^2;
% Neumann BCs via ghost points: u(0)=u(2), u(nx+1)=u(nx-1)
Lu(1)  = (u(2)     - 2*u(1)   + u(2))     / dx^2;
Lu(nx) = (u(nx-1)  - 2*u(nx)  + u(nx-1))  / dx^2;
end

function Du = upwind_first_derivative(u, V, dx)
% First derivative with upwind (handles scalar or species-specific V)
% u: 1×nx, V: scalar speed (>0 to +x, <0 to -x)
nx = numel(u);
Du = zeros(1,nx);
if V >= 0
    % backward difference (upwind for V>0)
    Du(2:nx) = (u(2:nx) - u(1:nx-1)) / dx;
    Du(1)    = 0; % zero gradient at boundary
else
    % forward difference (upwind for V<0)
    Du(1:nx-1) = (u(2:nx) - u(1:nx-1)) / dx;
    Du(nx)     = 0;
end
Du = V * Du;
end

function [I,E] = levy_jumps_space(I, E, t, lambdaJ, aJ, dt, tgt)
% Multiplicative Lévy jumps across space for pre system
% I: 3×nx, E: 1×nx
if t < tgt_window(tgt, true) || t > tgt_window(tgt, false) || lambdaJ<=0 || aJ<=0
    return;
end
% Number of events at this dt (shared for all nodes)
N = poissrnd(lambdaJ*dt);
if N == 0, return; end
% For each event, draw a multiplier (truncated normal) and apply to targets
for k = 1:N
    if tgt.I
        JI    = max(aJ*randn(3,1), -0.9);
        multI = (1 + JI);
        I     = I .* multI; % broadcasts across nx
    end
    if tgt.E
        JE = max(aJ*randn, -0.9);
        E  = E .* (1 + JE);
    end
end
end

function [I1,I2,E] = levy_jumps_space_mixed(I1,I2,E, t, lambdaJ, aJ, dt, tgt)
% Multiplicative Lévy jumps across space for mixed system
if t < tgt_window(tgt, true) || t > tgt_window(tgt, false) || lambdaJ<=0 || aJ<=0
    return;
end
N = poissrnd(lambdaJ*dt);
if N == 0, return; end
for k = 1:N
    if all(tgt.I1)
        JI = max(aJ*randn(3,1), -0.9);
        I1 = I1 .* (1 + JI);
    end
    if all(tgt.I2)
        JI = max(aJ*randn(3,1), -0.9);
        I2 = I2 .* (1 + JI);
    end
    if tgt.E
        JE = max(aJ*randn, -0.9);
        E  = E .* (1 + JE);
    end
end
end

function bd = tgt_window(tgt, isStart)
% Helper to supply the pre/post jump window bounds.
% We pass the struct itself to infer whether this call is for pre or post.
% For simplicity (no global state), we hard-code the windows to match top.
persistent tJ0_pre tJ1_pre tJ0_post tJ1_post
if isempty(tJ0_pre)
    tJ0_pre  = 10;   tJ1_pre  = 30;
    tJ0_post = 1600; tJ1_post = 2000;
end
% crude heuristic: presence of fields I1/I2 => post window
isPost = isfield(tgt,'I1') || isfield(tgt,'I2');
if isPost
    bd = tJ0_post; if ~isStart, bd = tJ1_post; end
else
    bd = tJ0_pre;  if ~isStart, bd = tJ1_pre;  end
end
end
