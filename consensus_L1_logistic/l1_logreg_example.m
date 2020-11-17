% L1 regularized logistic regression (not distributed)

%% Generate problem data

% rand('seed', 0);
% randn('seed', 0);

close all;

rng(10);

logisticfun = @(x) 1./(1+exp(-x));

n = 50; 
m = 200;

w = sprandn(n, 1, 0.5);  % N(0,1), 10% sparse
v = randn(1);            % random intercept

X     = sprandn(m, n, 10/n);
y     = (X*w + v)>= 0;

b_true = sign(X*w + v);

% noise is function of problem size use 0.1 for large problem
b = sign(X*w + v + sqrt(0.1)*randn(m,1)); % labels with noise

A = spdiags(b, 0, m, m) * X;

ratio = sum(b == 1)/(m);
mu    = 0.1 * 1/m * norm((1-ratio)*sum(A(b==1,:),1) + ratio*sum(A(b==-1,:),1), 'inf');

x_true = [v; w];

%% Solve problem

[x_admm, history_admm]       = l1_logreg(A, b, mu, 1.0, 1.0);

if 0
    step_size__x = 0.025; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = 0.0; %0.025; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
elseif 1
    step_size__x = 0; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = 0.025; %0.025; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
elseif 0
    step_size__x = -0.0025; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = 0.025; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
elseif 1
    step_size__x = 0.0025; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = 0.025; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
elseif 0
    step_size__x = 0.025; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = -0.001; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
elseif 0 % This is impossibility check: as expected it will not work
    step_size__x = -0.001; %-0.0001; %[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001].';
    step_size__z = -0.01; %[0.025, 0.025, 0.025, 0.025, 0.025, 0.025].';
end

[x_topadmm, history_topadmm] = l1_logreg_top_admm(A, b, mu/100, 1.25, 1.25, step_size__x, step_size__z);
[x_topadmm_, history_topadmm_] = l1_logreg_top_admm__neg_dual_var(A, b, mu/100, 1.25, 1.25, step_size__x, step_size__z);


b_admm     = (X*x_admm(2:end) + x_admm(1));
b_topadmm  = (X*x_topadmm(2:end) + x_topadmm(1));
b_topadmm_ = (X*x_topadmm_(2:end) + x_topadmm_(1));

% compute the output (class assignment) of the model for each data point
y_true     = logisticfun(b_true) > 0.5;
y_admm     = logisticfun(b_admm) > 0.5;
y_topadmm  = logisticfun(b_topadmm) > 0.5;
y_topadmm_ = logisticfun(b_topadmm_) > 0.5;
% calculate percent correct (percentage of data points% that are correctly classified by the model)
pctcorrec_true     = sum(y_true==y) / length(y) * 100;
pctcorrec_admm     = sum(y_admm==y) / length(y) * 100;
pctcorrec_topadmm  = sum(y_topadmm==y) / length(y) * 100;
pctcorrec_topadmm_ = sum(y_topadmm_==y) / length(y) * 100;


figure(1); hold all; plot(x_true, 'd'); plot(x_admm, 'o'); plot(x_topadmm, '*'); plot(x_topadmm_, 's');  legend('true', 'admm', 'top\_admm', 'top\_admm\_neg\_dual')
%figure(2); hold all; plot(btrue, 'd'); plot(b_admm, 'o'); plot(b_topadmm, '*'); plot(b_topadmm_, 's');  legend('true', 'admm', 'top\_admm', 'top\_admm\_neg\_dual')
%% Reporting

K_topadmm  = length(history_topadmm.objval);
K_topadmm_ = length(history_topadmm_.objval);
K_admm     = length(history_admm.objval);

h = figure(3);
plot(1:K_topadmm, history_topadmm.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2); hold on;
plot(1:K_topadmm_, history_topadmm_.objval, 'g-.', 'MarkerSize', 10, 'LineWidth', 2);
plot(1:K_admm, history_admm.objval, 'r--', 'MarkerSize', 10, 'LineWidth', 2);
legend({'TOP-ADMM'; 'TOP-ADMM (neg dual)'; 'ADMM'});
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');

g = figure(4);
subplot(2,1,1);
semilogy(1:K_topadmm, max(1e-8, history_topadmm.r_norm), 'k', ...
    1:K_topadmm, history_topadmm.eps_pri, 'k--',  'LineWidth', 2); hold on;
semilogy(1:K_topadmm_, max(1e-8, history_topadmm_.r_norm), 'g', ...
    1:K_topadmm_, history_topadmm_.eps_pri, 'g--',  'LineWidth', 2); 
semilogy(1:K_admm, max(1e-8, history_admm.r_norm), 'r', ...
    1:K_admm, history_admm.eps_pri, 'r--',  'LineWidth', 2);
ylabel('primal residual: ||r||_2');
legend({'TOP-ADMM'; 'TOP-ADMM (tol)'; 'TOP-ADMM (neg dual)'; 'TOP-ADMM (tol) (neg dual)'; 'ADMM'; 'ADMM (tol)'});

subplot(2,1,2);
semilogy(1:K_topadmm, max(1e-8, history_topadmm.s_norm), 'k', ...
    1:K_topadmm, history_topadmm.eps_dual, 'k--', 'LineWidth', 2); hold on;
semilogy(1:K_topadmm_, max(1e-8, history_topadmm_.s_norm), 'g', ...
    1:K_topadmm_, history_topadmm_.eps_dual, 'g--', 'LineWidth', 2);
semilogy(1:K_admm, max(1e-8, history_admm.s_norm), 'r', ...
    1:K_admm, history_admm.eps_dual, 'r--', 'LineWidth', 2);
ylabel('dual residual: ||s||_2'); xlabel('iter (k)');
%legend({'TOP-ADMM'; 'TOP-ADMM (tol)'; 'ADMM'; 'ADMM (tol)'});
legend({'TOP-ADMM'; 'TOP-ADMM (tol)'; 'TOP-ADMM (neg dual)'; 'TOP-ADMM (tol) (neg dual)'; 'ADMM'; 'ADMM (tol)'});


figure(7); clf;
X_full = full(X); hold all;
X_true_positive = X_full((y_true==0),:);
X_true_negative = X_full((y_true==1),:);

if 0

scatter3(X_full(:,1), X_full(:,2), 2*y_true-1, 'd'); %hold on;
scatter3(X_full(:,1), X_full(:,2), 2*y_admm-1, 'o');
scatter3(X_full(:,1), X_full(:,2), 2*y_topadmm-1, '*');

else
    hold all;
    gscatter(X_full(:,1), X_full(:,2), 2*y_true-1); %hold on;
    gscatter(X_full(:,1), X_full(:,2), 2*y_admm-1);
    gscatter(X_full(:,1), X_full(:,2), 2*y_topadmm-1);
    
end

% 
% K = length(history.objval);                                                                                                        
% 
% h = figure;
% plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2); 
% ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');
% 
% g = figure;
% subplot(2,1,1);                                                                                                                    
% semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
%     1:K, history.eps_pri, 'k--',  'LineWidth', 2); 
% ylabel('||r||_2'); 
% 
% subplot(2,1,2);                                                                                                                    
% semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
%     1:K, history.eps_dual, 'k--', 'LineWidth', 2);   
% ylabel('||s||_2'); xlabel('iter (k)'); 



%%

figure(10); clf;


% visualize the model% prepare a grid of points to evaluate the model at
ax      = axis;
xvals   = linspace(ax(1),ax(2),100);
yvals   = linspace(ax(3),ax(4),100);
[xx,yy] = meshgrid(xvals,yvals);
% construct regressor matrix
X = [xx(:) yy(:)];
X(:,end+1) = 1;

x_admm_new     =  x_admm(2:end); x_admm_new(end+1) = x_admm(1);     
x_topadmm_new  =  x_topadmm(2:end); x_topadmm_new(end+1) = x_topadmm(1); 
b_admm_new     = X*x_admm; %(X*x_admm(2:end) + x_admm(1));
b_topadmm_new  = X*x_topadmm; %(X*x_topadmm(2:end) + x_topadmm(1));


% evaluate model at the points (but don't perform the final thresholding)
outputimage_admm = reshape(logisticfun(b_admm_new),[length(yvals) length(xvals)]);
% visualize the image (the default coordinate system for images% is 1:N where N is the number of pixels along each dimension.% we have to move the image to the proper position we% accomplish this by setting XData and YData.)
h3 = imagesc(outputimage_admm,[0 1]);
% the range of the logistic function is 0 to 1
set(h3,'XData',xvals,'YData',yvals);
colormap(hot);
colorbar;
% visualize the decision boundary associated with the model% by computing the 0.5-contour of the image
[c4,h4] = contour(xvals,yvals,outputimage_admm,[.5 .5]);
set(h4,'LineWidth',3,'LineColor',[0 0 1]);
  % make the line thick and blue% send the image to the bottom so that we can see the data points
uistack(h3,'bottom')
% send the contour to the top
uistack(h4,'top');
% restore the original axis range
axis(ax);
% report the accuracy of the model in the title
title(sprintf('Classification accuracy is %.1f%%',pctcorrec_admm))

