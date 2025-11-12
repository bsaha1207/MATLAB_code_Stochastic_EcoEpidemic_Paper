% ========================= Spillover per Population Set =========================
% Uses your equations:
%   ED_i^k(x,t)=∫∫ J_i(x-y) h_i(t-τ) I_i^k(y,τ) dτ dy
%   ED_p^k(x,t)=∫∫ q(x-y)   f(t-τ) E^k(y,τ)   dτ dy
%   λ_k(x,t) = (1/Ψ_k) * (Σ_i θ_{k,i} ED_i^k + θ_{k,p} ED_p^k)
%   P_any,k(t) = 1 - exp(-∫_0^t ∫_Ω λ_k dx dt),   k ∈ {1 (NDC), 2 (DC)}

% ---------- PARAMETERS (set per set; replace with your empirical values) ----------
Psi1     = 1.0;                    % minimal infective dose (set 1)
Psi2     = 1.0;                    % minimal infective dose (set 2)
theta1_i = [0.7; 0.5; 0.4];        % human–species contact for set 1 (3x1)
theta2_i = [0.7; 0.5; 0.4];        % human–species contact for set 2 (3x1)
theta1_p = 0.6;                    % human–environment contact for set 1
theta2_p = 0.6;                    % human–environment contact for set 2

% Temporal survival kernels (exponential); use time-varying schedules if needed
tau_h = [20; 18; 22];              % infectiousness survival (species i)
tau_f = 30;                        % environmental infectivity survival

% Spatial kernels (examples = Gaussian; swap with your fitted kernels)
ell_J = [6; 8; 10];                % movement std for J_i (x-units)
ell_q = 12;                        % dispersal std for q

% ---------- Grid helpers ----------
Nt = Nt_full; nx = numel(x); L = Lx;
dx_periodic = min(abs(x - x(1)), L - abs(x - x(1)));   % periodic 1D distance to reference
gauss = @(d,ell) exp(-(d.^2)/(2*ell^2));
norm_by_trapz = @(k) k ./ trapz(x, k);

% Spatial kernels (normalized to integrate to 1) and FFTs
J_fft = cell(3,1);
for sp = 1:3
    Ji = norm_by_trapz( gauss(dx_periodic, ell_J(sp)) );
    J_fft{sp} = fft(Ji);
end
qi   = norm_by_trapz( gauss(dx_periodic, ell_q) );
q_fft = fft(qi);

% Exact discrete time-recursion coefficients for exponentials
a_h = exp(-dt ./ tau_h(:));               % 3x1
b_h = 1 - a_h;
a_f = exp(-dt /  tau_f);
b_f = 1 - a_f;

% ---------- Allocate per-set exposure doses ----------
EDi1 = zeros(Nt, nx, 3);   EDp1 = zeros(Nt, nx);  % set 1
EDi2 = zeros(Nt, nx, 3);   EDp2 = zeros(Nt, nx);  % set 2

% ---------- Time loop: spatial FFT-conv then temporal recursion ----------
for n = 1:Nt
    % ----- Set 1 (non-disease-carrying) -----
    for sp = 1:3
        I1_now = squeeze(I1f(n,:,sp)).';                      % nx x 1
        conv_I1 = real( ifft( fft(I1_now) .* J_fft{sp} ) );   % J_i * I1
        if n == 1
            EDi1(n,:,sp) = (b_h(sp)) * conv_I1.';
        else
            EDi1(n,:,sp) = a_h(sp) * squeeze(EDi1(n-1,:,sp)) + (b_h(sp)) * conv_I1.';
        end
    end
    E1_now = E1f(n,:).';
    conv_E1 = real( ifft( fft(E1_now) .* q_fft ) );           % q * E1
    if n == 1
        EDp1(n,:) = (b_f) * conv_E1.';
    else
        EDp1(n,:) = a_f * EDp1(n-1,:) + (b_f) * conv_E1.';
    end

    % ----- Set 2 (disease-carrying) -----
    for sp = 1:3
        I2_now = squeeze(I2f(n,:,sp)).';
        conv_I2 = real( ifft( fft(I2_now) .* J_fft{sp} ) );    % J_i * I2
        if n == 1
            EDi2(n,:,sp) = (b_h(sp)) * conv_I2.';
        else
            EDi2(n,:,sp) = a_h(sp) * squeeze(EDi2(n-1,:,sp)) + (b_h(sp)) * conv_I2.';
        end
    end
    E2_now = E2f(n,:).';
    conv_E2 = real( ifft( fft(E2_now) .* q_fft ) );           % q * E2 (post = shared)
    if n == 1
        EDp2(n,:) = (b_f) * conv_E2.';
    else
        EDp2(n,:) = a_f * EDp2(n-1,:) + (b_f) * conv_E2.';
    end
end

% ---------- Intensities λ_k(x,t) ----------
lambda1_xt = zeros(Nt, nx);
lambda2_xt = zeros(Nt, nx);
for n = 1:Nt
    dose1 = theta1_i(1)*EDi1(n,:,1) + theta1_i(2)*EDi1(n,:,2) + theta1_i(3)*EDi1(n,:,3) ...
            + theta1_p * EDp1(n,:);
    dose2 = theta2_i(1)*EDi2(n,:,1) + theta2_i(2)*EDi2(n,:,2) + theta2_i(3)*EDi2(n,:,3) ...
            + theta2_p * EDp2(n,:);
    lambda1_xt(n,:) = max(dose1 / Psi1, 0);
    lambda2_xt(n,:) = max(dose2 / Psi2, 0);
end

% ---------- Cumulative measures and probabilities ----------
Lambda1_t = zeros(Nt,1);   Lambda2_t = zeros(Nt,1);
for n = 2:Nt
    L1prime = trapz(x, lambda1_xt(n,:));
    L2prime = trapz(x, lambda2_xt(n,:));
    Lambda1_t(n) = Lambda1_t(n-1) + L1prime * dt;
    Lambda2_t(n) = Lambda2_t(n-1) + L2prime * dt;
end
P_any1 = 1 - exp(-Lambda1_t);       % set 1
P_any2 = 1 - exp(-Lambda2_t);       % set 2

% Spatially resolved P_any(x,t)
Lambda1_xt_cum = cumsum(lambda1_xt, 1) * dt;   % Nt x nx
Lambda2_xt_cum = cumsum(lambda2_xt, 1) * dt;   % Nt x nx
P_any1_xt = 1 - exp(-Lambda1_xt_cum);
P_any2_xt = 1 - exp(-Lambda2_xt_cum);

[Xgrid, Tgrid] = meshgrid(x, t_full);

% --- ED_i^1 (Set 1, species 1 only) ---
figure('Color','w');
surf(Xgrid, Tgrid, squeeze(EDi1(:,:,1)), 'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); zlabel('ED_i^(x,t)-N.D.C');
%title('Exposure Dose from Species 1 — Set 1 (NDC)');
%hold on; yline(t_perturb,'k--','LineWidth',1.2); hold off;

% --- ED_p^1 (Set 1, environment) ---
figure('Color','w');
surf(Xgrid, Tgrid, EDp1, 'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); zlabel('ED_p(x,t)-N.D.C');
%title('Environmental Exposure Dose — Set 1 (NDC)');
%hold on; yline(t_perturb,'k--','LineWidth',1.2); hold off;

% --- ED_i^2 (Set 2, species 1 only) ---
figure('Color','w');
surf(Xgrid, Tgrid, squeeze(EDi2(:,:,1)), 'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); zlabel('ED_i(x,t)-D.C.');
%title('Exposure Dose from Species 1 — Set 2 (DC)');
%hold on; yline(t_perturb,'k--','LineWidth',1.2); hold off;

% --- ED_p^2 (Set 2, environment) ---
figure('Color','w');
surf(Xgrid, Tgrid, EDp2, 'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); zlabel('ED_p(x,t)-D.C.');
%title('Environmental Exposure Dose — Set 2 (DC)');
%hold on; yline(t_perturb,'k--','LineWidth',1.2); hold off;

% --- Spillover Probabilities P_any(x,t) ---
figure('Color','w');
surf(Xgrid, Tgrid, P_any1_xt, 'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); zlabel('P_{\rm any}^{(1)}(x,t)');
title('Spillover Probability P_{any}(x,t) — Set 1 (NDC)');
hold on; yline(t_perturb,'k--','LineWidth',1.2); hold off;

figure('Color','w');
surf(Xgrid, Tgrid, P_any2_xt, 'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); zlabel('P_{\rm any}^{(2)}(x,t)');
title('Spillover Probability P_{any}(x,t) — Set 2 (DC)');
hold on; yline(t_perturb,'k--','LineWidth',1.2); hold off;
% ========================= Plots =========================
% (A) P_any for both sets
figure('Color','w'); 
plot(t_full, P_any1, 'LineWidth', 1.8); hold on;
plot(t_full, P_any2, 'LineWidth', 1.8);
xline(t_perturb,'k--','LineWidth',1.2); grid on;
xlabel('Time'); ylabel('P_{\rm any}([0,t]\times\Omega)');
title('Probability of \geq 1 Spillover by Time t (per population set)');
legend({'Set 1 (NDC)','Set 2 (DC)','t_{perturb}'}, 'Location','southeast'); hold off;

figure('Color','w');
plot( t_full, lambda1_xt(:,1), 'LineWidth', 1.8);hold on; 
plot( t_full, lambda2_xt(:,1), 'LineWidth', 1.8);hold on; 
xline(t_perturb,'k--','LineWidth',1.2,'DisplayName','perturb'); grid on; box on; 
xlabel('Time (months)'); ylabel('Cumulative hazard H(t)'); 
%title('Cumulative hazard H(t)'); 
legend('Location','best');

%% ===== Pre vs Post decomposition for P_any (per set) =====
Nt = numel(t_full); nx = numel(x);
n0 = find(t_full >= t_perturb, 1, 'first');   % index of the event time

% Time-cumulative intensity from t=0
Lambda1_cum = cumtrapz(t_full, lambda1_xt, 1);   % Nt x nx
Lambda2_cum = cumtrapz(t_full, lambda2_xt, 1);

% --- PRE: integrate only up to t_perturb, then hold constant afterwards
Lambda1_pre_xt = zeros(Nt, nx);
Lambda2_pre_xt = zeros(Nt, nx);
Lambda1_pre_xt(1:n0, :) = Lambda1_cum(1:n0, :);
Lambda2_pre_xt(1:n0, :) = Lambda2_cum(1:n0, :);
if n0 < Nt
    Lambda1_pre_xt(n0+1:end, :) = repmat(Lambda1_cum(n0, :), Nt-n0, 1);
    Lambda2_pre_xt(n0+1:end, :) = repmat(Lambda2_cum(n0, :), Nt-n0, 1);
end

% --- POST (windowed/conditional): subtract the pre mass at t_perturb
Lambda1_post_xt = zeros(Nt, nx);
Lambda2_post_xt = zeros(Nt, nx);
if n0 < Nt
    Lambda1_post_xt(n0:end, :) = Lambda1_cum(n0:end, :) - Lambda1_cum(n0, :);
    Lambda2_post_xt(n0:end, :) = Lambda2_cum(n0:end, :) - Lambda2_cum(n0, :);
end

% Probabilities
P_any1_pre_xt   = 1 - exp(-Lambda1_pre_xt);
P_any2_pre_xt   = 1 - exp(-Lambda2_pre_xt);
P_any1_post_xt  = 1 - exp(-Lambda1_post_xt);    % conditional/windowed (0 before t_perturb)
P_any2_post_xt  = 1 - exp(-Lambda2_post_xt);
P_any1_total_xt = 1 - exp(-Lambda1_cum);        % sanity: = 1 - exp(-(pre+post))
P_any2_total_xt = 1 - exp(-Lambda2_cum);

% Space-integrated curves (optional)
Lambda1_pre_t   = cumtrapz(t_full, trapz(x, lambda1_xt .* (t_full <= t_perturb), 2));
Lambda2_pre_t   = cumtrapz(t_full, trapz(x, lambda2_xt .* (t_full <= t_perturb), 2));
Lambda1_post_t  = zeros(Nt,1);  Lambda2_post_t = zeros(Nt,1);
if n0 < Nt
    L1post = trapz(x, lambda1_xt(n0:end,:), 2);
    L2post = trapz(x, lambda2_xt(n0:end,:), 2);
    Lambda1_post_t(n0:end) = cumtrapz(t_full(n0:end), L1post);
    Lambda2_post_t(n0:end) = cumtrapz(t_full(n0:end), L2post);
end
P_any1_pre  = 1 - exp(-Lambda1_pre_t);
P_any2_pre  = 1 - exp(-Lambda2_pre_t);
P_any1_post = 1 - exp(-Lambda1_post_t);
P_any2_post = 1 - exp(-Lambda2_post_t);
P_any1_tot  = 1 - exp(- (Lambda1_pre_t + Lambda1_post_t));
P_any2_tot  = 1 - exp(- (Lambda2_pre_t + Lambda2_post_t));

% ===== Plots (surfaces for pre, post, total) =====
[Xgrid, Tgrid] = meshgrid(x, t_full);

% Set 1 (NDC)
figure('Color','w'); surf(Xgrid,Tgrid,P_any1_total_xt,'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); zlabel('P^{(1)}_{\rm total}(x,t)');
title('TOTAL Spillover Probability — Set 1'); hold on; yline(t_perturb,'k--','LineWidth',1.2); hold off;

% Set 2 (DC)
figure('Color','w'); surf(Xgrid,Tgrid,P_any2_total_xt,'EdgeColor','none'); view(3); colormap(parula); colorbar;
xlabel('Space (x)'); ylabel('Time'); zlabel('P^{(2)}_{\rm total}(x,t)');
title('TOTAL Spillover Probability — Set 2'); hold on; yline(t_perturb,'k--','LineWidth',1.2); hold off;

