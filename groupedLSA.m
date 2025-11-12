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

