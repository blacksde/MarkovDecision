%%

% sim settings
T = 120;
K = 5; % state space of x = 0, 1, ..., K
% parameters
beta = 0.99;
RC = 11.7;
theta1 = 2.5;
c = @(x)(0.001*theta1*x);
%theta3 = [0.0937 0.4475 0.4459 0.0127 0.0002];
% nu is the utility without shoch
% nu(x, d) = -c(x) if d = 0
% nu(x, d) = -RC-c(x) if d = 1
nu =@(x,d)(-d*RC-c(x));
% transition density when d = 0(no replacement)
% x_{t+1} = max{x_t + 1, K}
F0 = diag(ones(K,1),1);
F0(K+1,K+1)=1;
% transition density when d = 1(with replacement)
% x_{t+1} = 0
F1 = [ones(K+1,1),zeros(K+1,K)];








%%