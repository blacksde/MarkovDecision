% alternative set of true parameters


addpath 'C:\Users\Jasmine\Documents\yzc\framework\MarkovDecision';
%%
param.runInitEV = 1;
param.nT = 120;
param.nM = 50;
param.nGrid = 2;  %number of states for each z_j
param.beta=0.95; % beta used
param.theta.nparam = 7;
param.theta.VP0 = 0.5;
param.theta.VP1 = 1.0;
param.theta.VP2 = -1.0;
param.theta.FC0 = 0.5;
param.theta.FC1 = 0.5;
param.theta.EC0 = 1.0;
param.theta.EC1 = 1.0;
param.theta.pnames = {'VP0','VP1','VP2','FC0','FC1','EC0','EC1'};
% Exogeneous state variable
param.min_z = 0;
param.max_z = 1;
param.min_omega = -1;
param.max_omega = 1;
param.gamma.z_0 = 0; %Parameters of z_j transition
param.gamma.z_1 = 0.6;
param.gamma.omega_0 = 0.2; % Parameters of omega transition(productivity)
param.gamma.omega_1 = 0.9;
param.gamma.sigma_z = 1;
param.gamma.sigma_omega = 1;

%system and MC parameters
param.MC=2;					  %number of MC iterations
param.multistarts=5;		%number of tries for each estimation


% Run tomLab
if ~exist("conAssign")
    run C:\tomlab\startup.m
else
    disp("TomLab Initiated");
end


%% True EV
%-------------------------------------------------------------------------
% Example: Solve for equilibrium
%-------------------------------------------------------------------------
theta = param.theta;
theta_vec0 = [param.theta.VP0,param.theta.VP1,param.theta.VP2,param.theta.FC0,param.theta.FC1,param.theta.EC0,param.theta.EC1]';
[P,state] = EntryExit.statetransition(param);

%%
%-------------------------------------------------------------------------
% Compare the converged CCP of Value Function iteration and EE iteration
%-------------------------------------------------------------------------
tic;
[ev,p1] = EntryExit.solve(P,state,param.beta,param.theta);
toc;

tic;
[v_til,p1_ee] = EulerEquation.solve(P,state,param.beta,theta_vec0);
toc;

%%
%-------------------------------------------------------------------------
% Simulate data
%-------------------------------------------------------------------------
if param.runInitEV == 1
    [data.MC_at,data.MC_yt,data.MC_zt,data.MC_zt_1] = simdata(P,state,p1,param);
    save EntryExitSimulate.mat;
else
    load('EntryExitSimulate.mat');
    disp('LOADING');
end
%%
%-------------------------------------------------------------------------
% Estimate likelihood using Full MLE
%-------------------------------------------------------------------------
S.N     = param.nM;
S.T     = param.nT;
S.beta  = param.beta;
S.P     = P;
S.state = state;
S.n_state = size(P,2);
S.n_action = 2;
%for i_mc=1:param.MC
i_mc = 1;
theta = param.theta;
	%convert simulated data to Bertel's format (? who's Bertel?)
	datasim.at=reshape(data.MC_at(:,:,i_mc),S.T*S.N,1);
	datasim.zt=reshape(data.MC_zt(:,:,i_mc),S.T*S.N,1);
    datasim.zt_1=reshape(data.MC_zt_1(:,:,i_mc),S.T*S.N,1);
	datasim.yt=reshape(data.MC_yt(:,:,i_mc),S.T*S.N,1);
    logl = DDCMixture.ll(datasim,theta_vec0,S);
%     logl = ll(datasim,theta,S);
    fprintf('The true likelihood is %.4f\n',sum(logl));
%end
%%
%-------------------------------------------------------------------------
% Check the consistency of exogeneous variable
%-------------------------------------------------------------------------
% a1 = tabulate(datasim.zt(datasim.zt_1 == 2))
% a2 = P(2,:)
% b1 = tabulate(datasim.at( (datasim.zt == 15 .* datasim.yt == 1 ) ))
% p1(15)
%%
%-------------------------------------------------------------------------
% Newton Method NFXP
%-------------------------------------------------------------------------
% theta0 = param.theta;
% Use Newton-Raphson method
ts = tic;
theta_vec0 = zeros(7,1);
llfun = @(theta_vec)DDCMixture.ll(datasim,theta_vec,S);
theta_newton = NFXP.estimation_newton(llfun, theta_vec0);
NW_NFXP_time = toc(ts) ;
disp(theta_newton);
%%
%-------------------------------------------------------------------------
% 2Step-EE
%-------------------------------------------------------------------------

S.N     = param.nM;
S.T     = param.nT;
S.beta  = param.beta;
S.P     = P;
S.state = state;
S.nstate = size(state,1);
%Both used Nlopt
ts = tic;
[theta_vec,p1] = EulerEquation.estimation_2step(theta_vec0,datasim,S);
TSTEP_EE_time = toc(ts);
disp(theta_vec);
%%
%-------------------------------------------------------------------------
% MPEC-EE
%-------------------------------------------------------------------------
ts = tic;
n_state = size(P,2);
[theta_vec_mpec,p1] = EulerEquation.estimation_mpec(theta_vec0,datasim,reshape(p1_ee,2*n_state,1),S) ;
MPEC_EE_time  = toc(ts);
%%
%--------------------------------------------------------------------------
% MPEC-NFXP
%--------------------------------------------------------------------------
theta_ev_vec = zeros(71,1);
theta_ev_vec = [theta_vec0;reshape(ev,2*n_state,1)] ;
cons = @(theta_ev_vec) NFXP.constraint(theta_ev_vec(1:7),theta_ev_vec(8:end),S);
d_cons = @(theta_ev_vec) NFXP.d_constraint(theta_ev_vec(1:7),theta_ev_vec(8:end),S);
obj  = @(theta_ev_vec) - NFXP.likelihood(theta_ev_vec(1:7),theta_ev_vec(8:end),datasim,S);
g    = @(theta_ev_vec) - NFXP.g_likelihood(theta_ev_vec(1:7),theta_ev_vec(8:end),datasim,S);
Name = 'The Problem';
n_state = size(S.state,1);
x_L = -inf * ones(7+2 * n_state,1); %Lower bound for parameters
x_U = inf * ones(7+2 * n_state,1); %Upper bound for parameters
c_L = zeros(2 * n_state,1);
c_U = zeros(2 * n_state,1);

Prob = conAssign(obj,g,[],[],[],[] ,Name,theta_ev_vec, ...
                   [], [], [], [], [], cons, [], [],[], c_L, c_U);
Prob.ConsDif = 1;
               % Prob.optParam.cTol = 1e-9;
ts = tic;
Result = tomRun('Knitro',Prob);
MPEC_NFXP_time = toc(ts);
PrintResult(Result, 2);

%%
fprintf('Newton-Raphson used %.8f seconds\n', NW_NFXP_time);
fprintf('2 step EE used %.8f seconds\n',TSTEP_EE_time);
fprintf('MPEC EE used %.8f seconds\n',MPEC_EE_time);
fprintf('MPEC NFXP used %.8f seconds\n',MPEC_NFXP_time);
