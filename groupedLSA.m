%% Local sensitivity of R0 (central difference, ±1%) with LaTeX bar plot
clc; clear;

% ===== Baseline parameters (your values) =====
n = 3;

D  = [0.2525; 0.101; 0.0505];         
V  = [0.202;  0.101; 0.0505];         

c  = [91.05000001; 76.02000001; 160.0000001];

B = [48.00000001, 12.00000001, 31.05; 
     60.00000001, 16.00000000, 0.0161;
     83.99999999, 43.99999999, 32.00000000];

beta = [ ...
    0.33,        0.394219246, 0.394219243; ...
    0.394219254, 0.33,        0.394219249; ...
    0.394219249, 0.394219249, 0.33];

alpha = [0.001085628; 0.001085628; 0.001085628];
gamma = [0.9; 0.9; 0.9];
mu    = [0.15;  0.15;  0.15];
rho   = [214.9757661; 214.9757681; 214.9757689];

delta   = 6;
sigma   = [0.650000001; 0.65; 0.649999999];
sigmaE  = 0.800000001;

gamma_jump = [0.650000001; 0.65; 0.649999999];
lambda_I   = [0.8; 0.8; 0.8];
lambda_E   = 0.650000001;

% ===== R0 as a function of parameters (exactly your derivation) =====
R0fun = @(S) compute_R0(S);

% Pack parameters in a struct for generic perturbation
S.D=D; S.V=V; S.c = c; S.B = B; S.beta = beta; S.alpha = alpha; S.gamma = gamma;
S.mu = mu; S.rho = rho; S.delta = delta; S.sigma = sigma; S.sigmaE = sigmaE;
S.gamma_jump = gamma_jump; S.lambda_I = lambda_I; S.lambda_E = lambda_E;

% Which fields to include in sensitivity (exclude D,V by request)
fields = {'c','B','beta','alpha','gamma','mu','rho','delta','sigma','sigmaE', ...
          'gamma_jump','lambda_I','lambda_E'};

% ===== Baseline R0 =====
R0_0 = R0fun(S);
fprintf('Baseline R0 = %.6f\n', R0_0);

% ===== Local sensitivity: central difference ±1% =====
rel = 0.01;  % 1% step
labels = {};
SI = [];     % elasticities
dR0 = [];    % raw derivatives
basevals = [];

for f = 1:numel(fields)
    name = fields{f};
    X = S.(name);
    for k = 1:numel(X)
        v = X(k);
        % step (use rel of |v|; if v≈0, step to 1% of 1 to avoid zero step)
        h = rel * v;

        Sp = S; Sm = S;
        Xp = X;  Xm = X;
        Xp(k) = v + h;
        Xm(k) = v - h;
        Sp.(name) = Xp;
        Sm.(name) = Xm;

        Rp = R0fun(Sp);
        Rm = R0fun(Sm);

        dR = (Rp - Rm)/(2*h);
        si  = dR * v / R0_0;

        dR0(end+1,1) = dR;          %#ok<SAGROW>
        SI(end+1,1)  = si;          %#ok<SAGROW>
        basevals(end+1,1) = v;      %#ok<SAGROW>
        labels{end+1,1} = latex_label(name, k, size(X)); %#ok<SAGROW>
    end
end

% ===== Table of results (sorted by |SI|) =====
T = table(labels, basevals, dR0, SI, 'VariableNames', ...
          {'Parameter','Baseline','dR0_dp','SI'});
[~,order] = sort(abs(T.SI), 'descend');
T = T(order,:);

disp(T(1:min(25,height(T)),:)); % show top 15 most sensitive parameters

% ===== Bar plot =====
figure('Color','w');
bar(T.SI, 'LineWidth', 0.8);
grid on; box off;

ax = gca;
ax.YGrid = 'on'; ax.XGrid = 'off';
ax.TickLength = [0 0];
ax.TickLabelInterpreter = 'latex';

xticks(1:height(T));
xticklabels(T.Parameter);
xtickangle(45);
ylabel('Sensitivity Index (elasticity)','Interpreter','latex');
xlabel('Parameter','Interpreter','latex');
title('Local Sensitivity of $R_0$ (central difference, $\pm 1\%$)','Interpreter','latex');

% ===== Helper: your exact R0 computation =====
function R0 = compute_R0(S)
    % Unpack
    c = S.c; B = S.B; beta = S.beta; alpha = S.alpha; rho = S.rho;
    gamma = S.gamma; mu = S.mu; delta = S.delta; sigma = S.sigma;
    sigmaE = S.sigmaE; gamma_jump = S.gamma_jump;
    lambda_I = S.lambda_I; lambda_E = S.lambda_E;

    % DFE
    Sstar = B \ c;

    % Jump penalties
    jump_core_I = gamma_jump - log(1 + gamma_jump);   % elementwise
    J_I = lambda_I .* jump_core_I;
    gE  = mean(gamma_jump);
    J_E = lambda_E * (gE - log(1 + gE));

    % Noise + jump–adjusted removals
    nu    = gamma + mu;
    t_nu  = nu + 0.5*(sigma.^2) + J_I;         % 3x1
    t_del = delta + 0.5*(sigmaE^2) + J_E;      % scalar

    % K_eff and R0
    K  = diag(Sstar) * ( beta + (alpha * (rho.'))/t_del ) * diag(1./t_nu);
    eigK = eig(K);
    R0 = max(abs(eigK));
end

% ===== Helper: build LaTeX label like \beta_{12}, \alpha_{3}, c_{2}, \sigma_E, \lambda_{I,1}, \gamma_{\mathrm{jump},1}
function lab = latex_label(fname, k, sz)
    % Map base names to LaTeX symbols
    baseMap = containers.Map( ...
        {'alpha','beta','gamma','mu','rho','delta','sigma','sigmaE', ...
         'lambda_I','lambda_E','c','B','gamma_jump'}, ...
        {'\alpha','\beta','\gamma','\mu','\rho','\delta','\sigma','\sigma', ...
         '\lambda','\lambda','c','b','\gamma'} );

    if ~isKey(baseMap, fname)
        % Fallback: plain name with (optional) index
        if prod(sz)==1
            lab = ['$' fname '$'];
        elseif numel(sz)==2
            [i,j] = ind2sub(sz, k);
            lab = ['$' fname '_{' num2str(i) num2str(j) '}$'];
        else
            lab = ['$' fname '_{' num2str(k) '}$'];
        end
        return
    end

    symb = baseMap(fname);

    % Scalars with special subscripts
    if strcmp(fname,'delta')
        lab = ['$' symb '$'];                 % \delta
        return;
    elseif strcmp(fname,'sigmaE')
        lab = ['$' symb '_{E}$'];             % \sigma_E
        return;
    elseif strcmp(fname,'lambda_E')
        lab = ['$' symb '_{E}$'];             % \lambda_E
        return;
    end

    % Vectors
    if isvector_size(sz)
        i = k;
        switch fname
            case 'lambda_I'
                lab = ['$' symb '_{I,' num2str(i) '}$'];      % \lambda_{I,i}
            case 'gamma_jump'
                lab = ['$' symb '_{\mathrm{j},' num2str(i) '}$'];  % \gamma_{\mathrm{jump},i}
            otherwise
                lab = ['$' symb '_{' num2str(i) '}$'];        % \alpha_i, \gamma_i, \mu_i, \rho_i, \sigma_i, c_i
        end
        return;
    end

    % Matrices (e.g., beta, B)
    if numel(sz)==2
        [i,j] = ind2sub(sz, k);
        lab = ['$' symb '_{' num2str(i) num2str(j) '}$'];     % \beta_{ij}, B_{ij}
    else
        % Generic fallback
        lab = ['$' symb '_{' num2str(k) '}$'];
    end
end

function tf = isvector_size(sz)
    % True if size is 1xN or Nx1
    tf = (numel(sz)==2) && (sz(1)==1 || sz(2)==1);
end


%% ===== Split into two bar plots: first 25 and last 26 =====
N = height(T);
n1 = min(30, N);              % first panel count
idx1 = 1:n1;
idx2 = (n1+1):N;

% ---- Figure 1: first 25 (largest |SI|) ----
figure('Color','w');
bar(T.SI(idx1), 'LineWidth', 0.8);
grid on;

ax = gca;
ax.YGrid = 'on'; ax.XGrid = 'off';
ax.TickLength = [0 0];
ax.TickLabelInterpreter = 'latex';

xticks(1:numel(idx1));
xticklabels(T.Parameter(idx1));
xtickangle(45);

ylabel('Sensitivity Index (elasticity)','Interpreter','latex');
xlabel('Parameter','Interpreter','latex');
%title('Local Sensitivity Index — Top 25 by $|\mathrm{SI}|$','Interpreter','latex');

% ---- Figure 2: remaining (last 26) ----
if ~isempty(idx2)
    figure('Color','w');
    bar(T.SI(idx2), 'LineWidth', 0.8);
    grid on;

    ax = gca;
    ax.YGrid = 'on'; ax.XGrid = 'off';
    ax.TickLength = [0 0];
    ax.TickLabelInterpreter = 'latex';

    xticks(1:numel(idx2));
    xticklabels(T.Parameter(idx2));
    xtickangle(45);

    ylabel('Sensitivity Index (elasticity)','Interpreter','latex');
    xlabel('Parameter','Interpreter','latex');
    %title(sprintf('Local Sensitivity of $R_0$', numel(idx2)), 'Interpreter','latex');
end
%%following is the actual code for local sensitivity bar
%% ===== Group identical sensitivity indices and relabel as θ_a, θ_b, ... =====
tol = 1e-2;   % equality tolerance

% Sort by |SI| (desc) to keep informative order
[~, ordImpact] = sort(abs(T.SI), 'descend');
Ts = T(ordImpact, :);

N = height(Ts);
used = false(N,1);

groupVals   = [];     % SI per displayed bar
groupLabels = {};     % tick label per displayed bar ($\theta_a$ or original)
thetaLists  = {};     % members for caption (cell array of cellstrs)
thetaTags   = {};     % '\theta_a', '\theta_b', ...
thetaCount  = 0;

for i = 1:N
    if used(i), continue; end

    % indices equal to Ts.SI(i) within tolerance
    idx = find(abs(Ts.SI - Ts.SI(i)) <= tol);
    idx = idx(~used(idx));

    if numel(idx) == 1
        % Unique value -> keep original LaTeX label
        groupVals(end+1,1)   = Ts.SI(i);                %#ok<SAGROW>
        groupLabels{end+1,1} = Ts.Parameter{i};         %#ok<SAGROW>
        thetaTags{end+1,1}   = '';                      %#ok<SAGROW>
        thetaLists{end+1,1}  = {};                      %#ok<SAGROW>
    else
        % Multiple identical values -> create a new theta group
        thetaCount = thetaCount + 1;
        tag = sprintf('$\\theta_{%s}$', theta_suffix(thetaCount));

        groupVals(end+1,1)   = Ts.SI(i);                %#ok<SAGROW>
        groupLabels{end+1,1} = tag;                     %#ok<SAGROW>
        thetaTags{end+1,1}   = tag;                     %#ok<SAGROW>
        thetaLists{end+1,1}  = Ts.Parameter(idx);       %#ok<SAGROW>
    end

    used(idx) = true;
end

% ===== Plot grouped bar chart =====
figure('Color','w');
bar(groupVals, 'LineWidth', 0.8);
grid on;

ax = gca;
ax.YGrid = 'on'; ax.XGrid = 'off';
ax.TickLength = [0 0];
ax.TickLabelInterpreter = 'latex';

xticks(1:numel(groupVals));
xticklabels(groupLabels);
xtickangle(35);

ylabel('Sensitivity Index (elasticity)','Interpreter','latex');
xlabel('Parameter / group','Interpreter','latex');
title('Local Sensitivity of $R_0$ with identical indices grouped as $\theta$','Interpreter','latex');

% ===== Print LaTeX-ready mapping for caption =====
fprintf('\n%% --- LaTeX-ready mapping for the figure caption ---\n');
for k = 1:numel(thetaTags)
    if isempty(thetaTags{k}), continue; end   % skip unique labels
    members = string(thetaLists{k});
    joined  = strjoin(members, ', ');
    fprintf('%s (SI = %.6g): \\{%s\\}\n', thetaTags{k}, groupVals(k), joined);
end
fprintf('%% --- End mapping ---\n');

% ===== Helper: a,b,...,z,aa,ab,... for theta subscripts =====
function s = theta_suffix(k)
    letters = 'abcdefghijklmnopqrstuvwxyz';
    s = "";
    while k > 0
        k = k - 1;
        s = letters(mod(k,26)+1) + s;
        k = floor(k/26);
    end
end

