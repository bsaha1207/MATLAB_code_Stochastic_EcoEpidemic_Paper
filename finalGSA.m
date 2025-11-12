%% Bar plot: params (x) vs Importance (col 2), keep >=5% normalized
clear; clc;

file = 'CRT_variable_importance2.xlsx';
T = readtable(file, 'PreserveVariableNames', true);

% Drop a duplicated header row if present
if (ischar(T{1,1}) || isstring(T{1,1}))
    s1 = lower(strtrim(string(T{1,1})));
    s2 = lower(strtrim(string(T{1,2})));
    if contains(s1,'independent') && contains(s2,'importance')
        T(1,:) = [];
    end
end

% Columns: 1=name, 2=importance
params = string(T{:,1});
imp    = str2double(string(T{:,2}));

% Normalized importance (%) from column 2 and filter
normImp = 100 * imp ./ max(imp);
keep = isfinite(imp) & normImp >= 5;
params = params(keep);
imp    = imp(keep);
normImp = normImp(keep);

% Sort by normalized importance (desc)
[~, idx] = sort(normImp, 'descend');
params = params(idx);
imp    = imp(idx);

% ----- Build LaTeX labels -----
latexLabels = arrayfun(@(s) localLatexify(char(s)), params, 'UniformOutput', false);

% ----- Plot -----
figure('Color','w');
bar(imp, 'FaceColor', [0.30 0.55 0.80], 'LineWidth', 0.8);
grid on;

ax = gca;
ax.YGrid = 'on'; ax.XGrid = 'off';
ax.TickLength = [0 0];                 % remove little tick marks
ax.TickLabelInterpreter = 'latex';     % render LaTeX

xticks(1:numel(params));
xticklabels(latexLabels);
xtickangle(45);

xlabel('Parameter','Interpreter','latex');
ylabel('Importance (column 2)','Interpreter','latex');
title('Parameter Importance (keeping $\geq$ 5\% normalized)','Interpreter','latex');

% ---------- Helper: convert names like alpha_3, beta_12, c3, b21 to LaTeX ----------
function out = localLatexify(s)
    % Map base tokens to LaTeX
    keys = {'alpha','beta','gamma','delta','sigma','rho','mu','lambda','c','b'};
    vals = {'\alpha','\beta','\gamma','\delta','\sigma','\rho','\mu','\lambda','c','b'};
    base = s; idx = '';

    if contains(s,'_')                       % e.g., alpha_3
        p = split(s,'_'); base = p{1}; idx = p{2};
    else                                     % e.g., c3 -> base='c', idx='3'
        t = regexp(s,'^(.*?)(\d+)$','tokens','once');
        if ~isempty(t), base = t{1}; idx = t{2}; end
    end

    k = find(strcmpi(base, keys), 1);
    if ~isempty(k), baseLatex = vals{k}; else, baseLatex = base; end

    if isempty(idx)
        out = ['$' baseLatex '$'];                 % e.g., $c$
    else
        out = ['$' baseLatex '_{' idx '}$'];       % e.g., $\alpha_{3}$
    end
end

