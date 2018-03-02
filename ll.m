function [logl,ev,score] = ll(datasim,theta,S)
    %----------------------------------------------------------------------
    % ll:   Function that solves partial MLE
    % SYNTAX: logl = ll(datasim,theta,S)
    %----------------------------------------------------------------------
    % INPUT:
    %   DATA:
    %       datasim.at   : Observed action
    %       datasim.zt   : Observed exogeneous state variables
    %       datasim.yt   : Observed endogeneous state variable
    %   Theta: The parameters
    %       theta.VP0 ;
    %       theta.VP1 ;
    %       theta.VP2 ;
    %       theta.FC0 ;
    %       theta.FC1 ;
    %       theta.EC0 ;
    %       theta.EC1 ;
    %   S:  Other first stage estimated
    %       S.N     : Number of markets
    %       S.T     : Number of time periods
    %       S.beta  : Discount factor
    %       S.STATE : All exogeneous state, each datasim.z corresponds to one
    %                 of the states
    %       S.P     : Transition density
    %----------------------------------------------------------------------
    % OUTPUT:
    %   logl: The likelihood of given (a_it,z_it,y_it)
    %   ev  : The expected value function
    %----------------------------------------------------------------------
    %%
    % First solve for the equilibrium EV
    a = reshape(datasim.at,S.N*S.T,1);
    z = reshape(datasim.zt,S.N*S.T,1);
    y = reshape(datasim.yt,S.N*S.T,1);
    [ev,p1,F] = EntryExit.solve(S.P,S.state,S.beta,theta);
    [n_state,d_state] = size(S.state);
    ccp = reshape(p1,n_state,2);
    ind = (y - 1) * n_state + z ;
    data_p1 = p1(ind);

    logl  = [a==1].*log(data_p1) + [a==2].*log(1 - data_p1);
    %%
    % Compute score function
    % Step 1: Compute the derivative of pi w.r.t to mp
    dvpdth   = exp(S.state(:,5)).* [ones(n_state,1),S.state(:,1),S.state(:,2)]; %Derivative of variable profit
    dfcdth   = [ones(n_state,1),S.state(:,3)];% Derivative of fixed cost
    decdth   = [ones(n_state,1),S.state(:,4)];
    % Step 2: Compute the derivative of contraction mapping w.r.t mp
    dpidth_1 =  [dvpdth,-dfcdth,zeros(size(decdth))];
    dpidth_0 =  [dvpdth,-dfcdth,-decdth];
    dtdth_1  =  ccp(:,1) .* dpidth_1 ; %dpi/dtheta when y = 1
    dtdth_0  =  ccp(:,2) .* dpidth_0 ; %dpi/dtheta when y = 0
    dtdth    = [dtdth_1;dtdth_0];
    dpidth   = [dpidth_1;dpidth_0];
    %%
    % This is different than what I've seen, But I don't care
    % Step 3: Compute the derivative of EV w.r.t theta
    dvdth   = F \ dtdth;
    dvdth_1 = dvdth(1:n_state,:);
    dvdth_0 = dvdth((1+n_state):(2*n_state),:);
    devdth  = S.P * ( dvdth_1 - dvdth_0 );
    % Step 4: Compute the derivative of log-likelihood w.r.t theta
    score    = bsxfun(@times,( (a == 1) - p1(ind)) , dpidth(ind,:) +...
              S.beta *  devdth(z,:));

end
