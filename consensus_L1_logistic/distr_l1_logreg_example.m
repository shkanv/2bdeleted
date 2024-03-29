% Distributed L1 regularized logistic regression that compares the
% performance of TOP-ADMM against ADMM.

% BASELINE SOURCE CODE REFERENCE: S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, “Distributed
% optimization and statistical learning via the alternating direction method
% of multipliers,” Foundations and Trends® in Machine Learning, vol. 3,
% no. 1, pp. 1–122, 2011.
% https://web.stanford.edu/~boyd/papers/admm/logreg-l1/distr_l1_logreg_example.html

clear; 
%% 
select__top_admm_problem_reformulation = 'a'; % 'a'; 'b'; 'c'; 'd'; 'e' 


%% Generate problem data
rng('default');
rng(0);

logisticfun = @(x) 1./(1+exp(-x));

addpath(genpath(pwd));

n = 100; % feature length?
m = 200; % dimension of matrix Ai is m x n ?!
N = 100; % the number of examples in the matrix A?

w = sprandn(n, 1, 100/n);       % N(0,1), 10% sparse
v = randn(1);                  % random intercept

X0     = sprandn(m*N, n, 10/n);           % data / observations
b_true = sign(X0*w + v);
y      = (X0*w + v) > 0;

% noise is function of problem size use 0.1 for large problem
b0 = sign(X0*w + v + sqrt(0.1)*randn(m*N, 1)); % labels with noise

% packs all observations in to an m*N x n matrix
A0 = spdiags(b0, 0, m*N, m*N) * X0;

ratio = sum(b0 == 1)/(m*N);
mu    = (0.001)*1/(m*N) * norm((1-ratio)*sum(A0(b0==1,:),1) + ratio*sum(A0(b0==-1,:),1), 'inf');

x_true = [v; w];

%% Solve problem

% some predefined parameters
rho1         = 1;
rho2         = 0.01; 
gamma        = 1; 
alpha_relax  = 1; 

%% RUN CLASSICAL ADMM (that utilizes QUASI-NEWTON METHOD internally)

[x_admm, history_admm] = distr_l1_logreg(A0, b0, mu, N, rho1, alpha_relax);    


%% RUN TOP-ADMM 

switch lower(select__top_admm_problem_reformulation)
    case 'a'
    case 'b'
    case 'c'
    case 'd'
    case 'e'
    otherwise
        error('Unknown problem formulation');
end
if 0
    % works (slowest)
    step_size__x = 0.035; %75; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = 0.0; %0.025; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
    select_me    = 0;    
elseif 0
    % works (best)
    step_size__x = 0; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = 0.0025; %0.025; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
    select_me    = 0;
elseif 0
    % works (faster and best)
    step_size__x = -0.0005; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = 0.0025; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
    select_me    = 0;
elseif 1
    % works (ok)
    step_size__x = 0.075; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = -0.0001; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
    select_me    = 0;
elseif 1
    % works (faster and second best)
    step_size__x = 0.025; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = 0.0025; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
end



[x_topadmm, history_topadmm] = distr_l1_logreg_topadmm(A0, b0, mu, N, rho1, gamma, alpha_relax, step_size__x, step_size__z);


[x_topadmm_prox, history_topadmm_prox] = distr_l1_logreg_topadmm_proximal(A0, b0, mu, N, rho1, rho2, gamma, alpha_relax, step_size__x, step_size__z);


X               = X0;
b_admm          = (X*x_admm(2:end) + x_admm(1));
b_topadmm       = (X*x_topadmm(2:end) + x_topadmm(1));
b_topadmm_prox  = (X*x_topadmm_prox(2:end) + x_topadmm_prox(1));

% compute the output (class assignment) of the model for each data point
modelfit_true           = logisticfun(b_true) > 0.5;
modelfit_admm           = logisticfun(b_admm) > 0.5;
modelfit_topadmm        = logisticfun(b_topadmm) > 0.5;
modelfit_topadmm_prox   = logisticfun(b_topadmm_prox) > 0.5;

% calculate percent correct (percentage of data points% that are correctly classified by the model)
pctcorrec_true          = sum(modelfit_true==y) / length(y) * 100;
pctcorrec_admm          = sum(modelfit_admm==y) / length(y) * 100
pctcorrec_topadmm       = sum(modelfit_topadmm==y) / length(y) * 100
pctcorrec_topadmm_prox  = sum(modelfit_topadmm_prox==y) / length(y) * 100



%%
figure(1); clf; hold all; 
plot(x_true, 'd', 'markersize', 8); 
plot(x_admm, 'o', 'markersize', 8); 
plot(x_topadmm, '*', 'markersize', 8); 
plot(x_topadmm_prox, '<', 'markersize', 8);
ylabel('(Random) Input Data');
xlabel('Samples')
legend('TRUE', 'ADMM', sprintf('TOP-ADMM ( \\tau=%1.5f, \\vartheta=%1.5f )', step_size__z, step_size__x))

%% Reporting

K_topadmm      = length(history_topadmm.objval);
K_topadmm_prox = length(history_topadmm_prox.objval);
K_admm         = length(history_admm.objval);

h = figure(2); clf
plot(1:K_topadmm, history_topadmm.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2); hold on;
plot(1:K_topadmm_prox, history_topadmm_prox.objval, 'b-.', 'MarkerSize', 10, 'LineWidth', 2); 
plot(cumsum(mean(history_admm.LBFGS_iters,1)), history_admm.objval, 'r--', 'MarkerSize', 10, 'LineWidth', 2);
legend({'TOP-ADMM'; 'RTOP-ADMM'; sprintf('ADMM (avg LBFGS iters %1.2f)', sum(mean(history_admm.LBFGS_iters,1)))});
ylabel('$\sum_m h_m(x_m^k) + g(z^k)$'); xlabel('iter ($k$)');

g = figure(3); clf;
subplot(2,1,1);
semilogy(1:K_topadmm, max(1e-8, history_topadmm.r_norm), 'k', ...
    1:K_topadmm, history_topadmm.eps_pri, 'k--',  'LineWidth', 2); hold on;
semilogy(1:K_topadmm_prox, max(1e-8, history_topadmm_prox.r_norm), 'b', ...
    1:K_topadmm, history_topadmm_prox.eps_pri, 'b-.',  'LineWidth', 2);
semilogy(cumsum(mean(history_admm.LBFGS_iters,1)), max(1e-8, history_admm.r_norm), 'r', ...
    cumsum(mean(history_admm.LBFGS_iters,1)), history_admm.eps_pri, 'r--',  'LineWidth', 2);
ylabel('primal residual: ||r||_2');
legend({'TOP-ADMM'; 'TOP-ADMM (tol)'; 'RTOP-ADMM'; 'RTOP-ADMM (tol)'; 'ADMM'; 'ADMM (tol)'});
ylim([10e-5, 10e2]);
%xlim([0, 1000]);

subplot(2,1,2);
semilogy(1:K_topadmm, max(1e-8, history_topadmm.s_norm), 'k', ...
    1:K_topadmm, history_topadmm.eps_dual, 'k--', 'LineWidth', 2); hold on;
semilogy(1:K_topadmm_prox, max(1e-8, history_topadmm_prox.s_norm), 'b', ...
    1:K_topadmm, history_topadmm_prox.eps_dual, 'b-.', 'LineWidth', 2);
semilogy(cumsum(mean(history_admm.LBFGS_iters,1)), max(1e-8, history_admm.s_norm), 'r', ...
    cumsum(mean(history_admm.LBFGS_iters,1)), history_admm.eps_dual, 'r--', 'LineWidth', 2);
ylabel('dual residual: ||s||_2'); xlabel('iter (k)');
legend({'TOP-ADMM'; 'TOP-ADMM (tol)'; 'RTOP-ADMM'; 'RTOP-ADMM (tol)'; 'ADMM'; 'ADMM (tol)'});
ylim([10e-5, 10e2]);
%xlim([0, 1000]);


%%



