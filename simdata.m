function [MC_at,MC_yt,MC_zt,MC_zt_1] = simdata(P,state,p1,param)
    %---------------------------------------------------------------------    
    % SYNTAX: [MC_at,MC_yt,MC_zt,MC_zt_1] = simdata(P,state,p1, param)
    % INPUT: param
    %      P:   transition of exogenous data
    %      MC:  number of montecarlo
    %---------------------------------------------------------------------
    rand_seed = 100;
    rand('seed',rand_seed);
    %---------------------------------------------------------------------
    %** Set parameters
    %---------------------------------------------------------------------
    MC=param.MC;    %number of MC simulation
    nT = param.nT; %number of time periods
    nM = param.nM ;	%number of markets
    n_state = size(state,1); %number of exogenous states
    p1 = reshape(p1,n_state,2);
    tmp11= P .* repmat(p1(:,1),1,n_state); %y = 1, a = 1
    tmp10= P .* repmat((1 - p1(:,1)),1,n_state); %y = 1, a = 0
    tmp01= P .* repmat(p1(:,2),1,n_state);%y = 0, a = 1
    tmp00= P .* repmat((1 - p1(:,2)),1,n_state);%y = 0, a = 0
    F_x  = [tmp11 , tmp10 ; tmp01, tmp00]; %joint transition of (y,z)
    %% ---------------------------------------------------------------------
    %** Data structure
    %        MC_at:     record  a_t for t=1:nT
    %        MC_yt:     record  y_{t-1} for t=1:nT
    %        MC_zt:     record  z_t for t=1:nT
    %        MC_zt_1:   record  z_{t-1} for t=1:nT
    %---------------------------------------------------------------------
    MC_at = zeros(nM, nT, MC);
    MC_yt = zeros(nM, nT, MC);
    MC_zt = zeros(nM, nT, MC);
    MC_zt_1 = zeros(nM, nT, MC); %z t-1
    %% ---------------------------------------------------------------------
    %** Create Markov Process of Exogeneous Variable
    % x_t = (y_t,z_t) 
    % need to simulate nT + 2 times
    % need a_t : t = 0 to nT
    %            y_t = a_t-1
    % need z_t : t = 0 to nT
    %---------------------------------------------------------------------
    x_mc  = dtmc(F_x); 
    for kk = 1:MC
        for m = 1:nM
           x_sim = simulate(x_mc,nT + 2);    
           a     = floor((x_sim - 1) / n_state) + 1;
           z     = rem(x_sim - 1, n_state)+1;
           MC_at(m,:,kk) = a(3:nT+2);
           MC_yt(m,:,kk) = a(2:nT+1);
           MC_zt(m,:,kk) = z(2:nT+1);
           MC_zt_1(m,:,kk) = z(1:nT);
        end
    end
    
end