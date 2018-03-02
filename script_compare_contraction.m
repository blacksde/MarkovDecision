%%
addpath 'C:\Users\Jasmine\Google Drive\miaomiaomiao!\miaomiao_project'
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
param.MC=250;					  %number of MC iterations
param.multistarts=5;		%number of tries for each estimation

%% True EV
%-------------------------------------------------------------------------
% Example: Solve for equilibrium
%-------------------------------------------------------------------------
theta = param.theta;
theta_vec0 = [param.theta.VP0,param.theta.VP1,param.theta.VP2,param.theta.FC0,param.theta.FC1,param.theta.EC0,param.theta.EC1]';
time_table = zeros(2,7);
diff_table = zeros(1,7);
size_table = zeros(1,7);
colNames = {'N2','N3','N4','N5'};
rowNames = {'Value Function','EE_p'};
%-------------------------------------------------------------------------
% Compare the converged CCP of Value Function iteration and EE iteration
%-------------------------------------------------------------------------
for i = 2:8
    param.nGrid = i;
    [P,state] = EntryExit.statetransition(param);
    n_state = size(P,2);
    size_table(i-1) = n_state;
    ts = tic;
    [ev,p1] = EntryExit.solve(P,state,param.beta,param.theta);
    time_table(1,i-1) = toc(ts);
    ts = tic;
    [v_til,p1_ee] = EulerEquation.solve(P,state,param.beta,theta_vec0);
    time_table(2,i-1) = toc(ts);
    diff_table(i-1) = max(abs(reshape(p1_ee,n_state*2,1)-p1));
end
disp(array2table(time_table,'RowNames',rowNames,'VariableNames',colNames));
disp(array2table(size_table,'RowNames',{'State Space Size'},'VariableNames',colNames));