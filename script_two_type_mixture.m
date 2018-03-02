addpath 'C:\Users\Jasmine\Documents\yzc\framework\MarkovDecision';
runInitEV = 1;
%%
param.nMC = 5;
param.nT = 120;
param.nM = 50;	
param.nGrid = 2;  %number of states for each z_j
param.beta=0.95; % beta used

% Exogeneous state variable that determine the exogenous transition
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

% Two thetas
theta1.nparam = 7;
theta1.VP0 = 0.5;
theta1.VP1 = 1.0;
theta1.VP2 = -1.0;
theta1.FC0 = 0.5;
theta1.FC1 = 1.0;
theta1.EC0 = 1.0;
theta1.EC1 = 1.0;
theta1.pnames = {'VP0','VP1','VP2','FC0','FC1','EC0','EC1'};

% theta2.VP0 = 0.5;
% theta2.VP1 = 1.0;
% theta2.VP2 = -1.0;
% theta2.FC0 = 0.5;
% theta2.FC1 = 1.5;
% theta2.EC0 = 1.0;
% theta2.EC1 = 1.0;
% theta2.pnames = {'VP0','VP1','VP2','FC0','FC1','EC0','EC1'};


%system and MC parameters
param.MC=1;					  %number of MC iterations

%parameters that determine the mixture
mix_param.nType = 2;
mix_param.mixProb = [0.5,0.5];

% Run tomLab
if ~exist("conAssign")
    run C:\tomlab\startup.m
else
    disp("TomLab Initiated");
end

%%
%The exogenous transition densities are the same
[P,state] = NFXP.statetransition(param);
%Assume the first stage parameters are known
S1.N     = param.nM;
S1.T     = param.nT;
S1.beta  = param.beta;
S1.P     = P;
S1.state = state;
S1.n_type = 2;
S1.n_action = 2;
S1.n_state = size(P,2);
theta_vec = [theta1.VP0,theta1.VP1,theta1.VP2,theta1.FC0,theta1.FC1,theta1.EC0,theta1.EC1]';
S2 = S1;
S2.state(:,5) = S1.state(:,5) -1 ; 
%--------------------------------------------------------------------------
%Generate Data
%--------------------------------------------------------------------------
%%
ResultTable_NFXP = zeros(param.nMC,7);
TimeTable_NFXP   = zeros(param.nMC,1);
IterTable_NFXP   = zeros(param.nMC,1);
ResultTable_EE   = zeros(param.nMC,7);
TimeTable_EE     = zeros(param.nMC,1);
IterTable_EE   = zeros(param.nMC,1);

ts = tic;
for i = 1: param.nMC
    [datasim.at,datasim.yt,datasim.zt] = ...
        DDCMixture.simdata_mix(S1,S2,theta_vec,param,mix_param);
    Data{i} = datasim;
end
TimeSimulation = toc(ts);
fprintf('Simulation of mixture data used %f seconds \n', TimeSimulation);
%%
%--------------------------------------------------------------------------
%Estimation in NFXP
%--------------------------------------------------------------------------

for i = 1:param.nMC
    
    datasim = Data{i};
    
    theta_vec0 = 0.25 * randn(7,1);
    
    %     ************** NFXP Estimation *****************************
    ts = tic;
    opt.method = 'NFXP';
    [theta_hat,w,iter] = DDCMixture.SequentialEstimation(datasim,mix_param,S1,S2,theta_vec0,opt);
    TimeEstimation = toc(ts);
    
    ResultTable_NFXP(i,:) = theta_hat ;
    TimeTable_NFXP(i) = TimeEstimation;
    IterTable_NFXP(i) = iter;
    %     ************** EE Estimation *****************************
    ts = tic;
    opt.method = 'EE';
    [theta_hat,w,iter] = DDCMixture.SequentialEstimation(datasim,mix_param,S1,S2,theta_vec0,opt);
    TimeEstimation =  toc(ts);
    ResultTable_EE(i,:) = theta_hat;
    IterTable_EE(i) = iter;
    TimeTable_EE(i) = TimeEstimation;
    fprintf('Estimating sample %d out of %d\n', i, param.nMC);
end

%%

disp('The average time using NFXP in 500 simulations');
disp( mean(TimeTable_NFXP));

disp('The average time using EE in 500 simulations');
disp( mean(TimeTable_EE));

disp('The average deviation using NFXP in 500 simulations');
disp( mean(ResultTable_NFXP));

disp('The average deviation using EE in 500 simulations');
disp( mean(ResultTable_EE));

disp('The average deviation using NFXP in 500 simulations');
disp( mean(abs(ResultTable_NFXP)- repmat(theta_vec',param.nMC,1)));

disp('The average deviation using EE in 500 simulations');
disp( mean(abs(ResultTable_EE)- repmat(theta_vec',param.nMC,1)));

disp('The average iterations using NFXP in 500 simulations');
disp( mean(IterTable_NFXP));

disp('The average iterations using EE in 500 simulations');
disp( mean(IterTable_EE));