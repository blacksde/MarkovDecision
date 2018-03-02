classdef DDCMixture
    methods (Static)
        function [at,yt,zt] = simdata(theta_vec, S,nT,nM)
            %---------------------------------------------------------------------    
            % SYNTAX: [at,yt,zt,zt_1] = simdata(theta_vec, S,nT,nM)
            % INPUT: param
            %      P:   transition of exogenous data
            %      MC:  number of montecarlo
            %---------------------------------------------------------------------
            [ev,p1,F] = NFXP.solve(theta_vec,S);
            n_state = S.n_state; %number of exogenous states
            p1 = reshape(p1,n_state,2);
            tmp11= S.P .* repmat(p1(:,1),1,n_state); %y = 1, a = 1
            tmp10= S.P .* repmat((1 - p1(:,1)),1,n_state); %y = 1, a = 0
            tmp01= S.P .* repmat(p1(:,2),1,n_state);%y = 0, a = 1
            tmp00= S.P .* repmat((1 - p1(:,2)),1,n_state);%y = 0, a = 0
            F_x  = [tmp11 , tmp10 ; tmp01, tmp00]; %joint transition of (y,z)
            % ---------------------------------------------------------------------
            %** Data structure
            %        at:     record  a_t for t=1:nT
            %        yt:     record  y_{t-1} for t=1:nT
            %        zt:     record  z_t for t=1:nT
            %        zt_1:   record  z_{t-1} for t=1:nT
            %---------------------------------------------------------------------
            at = zeros(nM, nT);
            yt = zeros(nM, nT);
            zt = zeros(nM, nT);
            zt_1 = zeros(nM, nT); %z t-1
            % ---------------------------------------------------------------------
            %** Create Markov Process of Exogeneous Variable
            % x_t = (y_t,z_t) 
            % need to simulate nT + 2 times
            % need a_t : t = 0 to nT
            %            y_t = a_t-1
            % need z_t : t = 0 to nT
            %---------------------------------------------------------------------
            x_mc  = dtmc(F_x); 
            
            for m = 1:nM
                   x_sim = simulate(x_mc,nT + 1);    
                   a     = floor((x_sim - 1) / n_state) + 1;
                   z     = rem(x_sim - 1, n_state)+1;
                   at(m,:) = a(2:nT+1);
                   yt(m,:) = a(1:nT);
                   zt(m,:) = z(1:nT);
                   
            end
        end
        
        function [at,yt,zt,zt_1] = simdata_mix(S1,S2,theta_vec,param,mix_param)
            %---------------------------------------------------------------------    
            % SYNTAX: [at,yt,zt,zt_1] = simdata(P,state,p1, param)
            % INPUT: param
            %      P:   transition of exogenous data
            %      MC:  number of montecarlo
            %---------------------------------------------------------------------
            %Solve for type I and type II CCP
            %Simulate data
            pd = makedist('Multinomial','probabilities',mix_param.mixProb);
            ubs_type = random(pd,param.nM,1);
            mix_prob = tabulate(ubs_type);
            mix_prob = mix_prob(1,3)/100;
            rand_seed = 100;
            rand('seed',rand_seed);
            %---------------------------------------------------------------------
            %** Set parameters
            %---------------------------------------------------------------------
            
            [at1,yt1,zt1] = DDCMixture.simdata(theta_vec,S1,param.nT,round(param.nM * mix_prob));
            [at2,yt2,zt2] = DDCMixture.simdata(theta_vec,S2,param.nT,round(param.nM * (1 - mix_prob)));
            at = [at1;at2];
            yt = [yt1;yt2];
            zt = [zt1;zt2];
            
        end %simdata_mix
        
        %---------------------------------------------------------------------    
      
        function [ll,ev,score] = ll(datasim,theta_vec,S)
            %[ll,ev,score] = DDCMixture.ll(datasim,theta_vec,S)
            
            % Log likelihood in NFXP
            % First solve for the equilibrium EV
            a = reshape(datasim.at,S.N*S.T,1);
            z = reshape(datasim.zt,S.N*S.T,1);
            y = reshape(datasim.yt,S.N*S.T,1);
            [ev,p1,F] = NFXP.solve(theta_vec,S);
            n_state = S.n_state;
            ind = (y - 1) * n_state + z ;
            data_p1 = p1(ind);

            ll  = (a==1).*log(data_p1) + (a==2).*log(1 - data_p1);
%             ll  = reshape(ll,S.N,S.T);
            %MULTIPLY THE WEIGHT HERE
%             ll  = reshape(ll,S.N*S.T,1);
            
            [devdth,dpidth] = DDCMixture.devdth(S,F,p1);
            % Step 4: Compute the derivative of log-likelihood w.r.t theta
            score    = bsxfun(@times,( (a == 1) - p1(ind)) , dpidth(ind,:) +...
                      S.beta *  devdth(z,:));  
        end
        
        %---------------------------------------------------------------------    
        
        function [devdth,dpidth] = devdth(S,F,p1)
            % Compute score function
                n_state = S.n_state;
                ccp = reshape(p1,n_state,2);
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

                % This is different than what I've seen, But I don't care 
                % Step 3: Compute the derivative of EV w.r.t theta
                dvdth   = F \ dtdth; 
                dvdth_1 = dvdth(1:n_state,:);
                dvdth_0 = dvdth((1+n_state):(2*n_state),:);
                devdth  = S.P * ( dvdth_1 - dvdth_0 );
        end
        
        function dpidth = dpidth(S)
            n_state = S.n_state;
                % Step 1: Compute the derivative of pi w.r.t to mp
                dvpdth   = exp(S.state(:,5)).* [ones(n_state,1),S.state(:,1),S.state(:,2)]; %Derivative of variable profit
                dfcdth   = [ones(n_state,1),S.state(:,3)];% Derivative of fixed cost
                decdth   = [ones(n_state,1),S.state(:,4)];
                % Step 2: Compute the derivative of contraction mapping w.r.t mp
                dpidth_1 =  [dvpdth,-dfcdth,zeros(size(decdth))];
                dpidth_0 =  [dvpdth,-dfcdth,-decdth];
                dpidth   = [dpidth_1;dpidth_0];
                
        end
        
        
            
        
        %------------------------------------------------------------------
        %*************      LIKELIHOOD FUNCTIONS       ********************
        %------------------------------------------------------------------
        function [logl, p1,p2,ll_1,ll_2 ] =  ll_NFXP(theta_vec,datasim,w,p1,p2,S1,S2)
            %% GMM to update theta using Euler Equation
            % Used in two-step estimator of theta
            % This function gives sum of log-likelihood when fixing the p1(h-1)
                        a = reshape(datasim.at,S1.N*S1.T,1);
            z = reshape(datasim.zt,S1.N*S1.T,1);
            y = reshape(datasim.yt,S1.N*S1.T,1);
            [n_state,d_state] = size(S1.state);
            ind = (y - 1) * n_state + z ;
            ccp_1 = reshape(p1,n_state,2);
            ccp_2 = reshape(p2,n_state,2);
            
            pi_1       = DDCMixture.dpidth(S1) * theta_vec ;
            pi_2       = DDCMixture.dpidth(S2) * theta_vec ;
            
            tmp_1 = pi_1 + S1.beta * kron([1;1], S1.P) * ( log( 1- ccp_1(:,2)) - log( 1 - ccp_1(:,1)) );
            p1_1 = exp(tmp_1) ./ ( 1 + exp(tmp_1));
            data_p_1 = p1_1(ind);
            
            tmp_2 = pi_2 + S2.beta * kron([1;1], S2.P) * ( log( 1 - ccp_2(:,2)) - log( 1 - ccp_2(:,1)) );
            p1_2 = exp(tmp_2) ./ ( 1 + exp(tmp_2));
            data_p_2 = p1_2(ind);
               
            ll_1  = sum(reshape([a==1].*log(data_p_1) + [a==2].*log(1 - data_p_1),S1.N,S1.T),2);
            ll_2  = sum(reshape([a==1].*log(data_p_2) + [a==2].*log(1 - data_p_2),S1.N,S1.T),2);
            

            lik = w(:,1) .* ll_1 + w(:,2) .* ll_2 ;
            logl = sum(lik);

            if nargout > 1
                p1 = reshape(p1_1,n_state * 2,1);
                p2 = reshape(p1_2,n_state * 2,1);
            end
        end
        

        function [logl, p1,p2,ll_1,ll_2 ] =  ll_EE(theta_vec,datasim,w,p1,p2,S1,S2)
            %% GMM to update theta using Euler Equation
            % Used in two-step estimator of theta
            % This function gives sum of log-likelihood when fixing the p1(h-1)
            a = reshape(datasim.at,S1.N*S1.T,1);
            z = reshape(datasim.zt,S1.N*S1.T,1);
            y = reshape(datasim.yt,S1.N*S1.T,1);
            [n_state,d_state] = size(S1.state);
            ind = (y - 1) * n_state + z ;
            ccp_1 = reshape(p1,n_state,2);
            ccp_2 = reshape(p2,n_state,2);
            
            pi_1       = DDCMixture.dpidth(S1) * theta_vec ;
            pi_2       = DDCMixture.dpidth(S2) * theta_vec ;
            
            tmp_1 = pi_1 + S1.beta * kron([1;1], S1.P) * ( log(1- ccp_1(:,2)) - log(1 - ccp_1(:,1)) );
            p1_1 = exp(tmp_1) ./ ( 1 + exp(tmp_1));
            data_p_1 = p1_1(ind);
            
            tmp_2 = pi_2 + S2.beta * kron([1;1], S2.P) * ( log(1- ccp_2(:,2)) - log(1 - ccp_2(:,1)) );
            p1_2 = exp(tmp_2) ./ ( 1 + exp(tmp_2));
            data_p_2 = p1_2(ind);
               
            ll_1  = sum(reshape([a==1].*log(data_p_1) + [a==2].*log(1 - data_p_1),S1.N,S1.T),2);
            ll_2  = sum(reshape([a==1].*log(data_p_2) + [a==2].*log(1 - data_p_2),S1.N,S1.T),2);
            

            lik = w(:,1) .* ll_1 + w(:,2) .* ll_2 ;
            logl = sum(lik);

            if nargout > 1
                p1 = reshape(p1_1,n_state * 2,1);
                p2 = reshape(p1_2,n_state * 2,1);
            end
        end
        
        %------------------------------------------------------------------
        % **************      SEQUENTIAL ESTIMATION       *****************
        %------------------------------------------------------------------

        function [theta_hat,w,iter] = SequentialEstimation(datasim,mix_param,S1,S2,theta_vec0,opt)
            
            %STEP 1: guess the weight for the two type mixture
            w0 = [0.1 * ones(S1.N,1), 0.9 * ones(S1.N,1) ]; %The weight
            q = mean(w0); %The probability of each type
            
            opt.tol  = 1e-3;
            opt.max_iter = 100;
            opt.output = 0;
            diff      = 1;
            iter      = 0;
            if string(opt.method) == 'EE'
                fprintf('Solving using EE\n');
                [vtil_1,p1_0] = EulerEquation.solve(S1.P,S1.state,S1.beta,theta_vec0);
                [v_til2,p2_0] = EulerEquation.solve(S1.P,S2.state,S2.beta,theta_vec0);
                ll_function   = @DDCMixture.ll_EE;
            elseif string(opt.method) == 'NFXP'
                [ev_1,p1_0] = NFXP.solve(theta_vec0,S1);
                [ev_2,p2_0] = NFXP.solve(theta_vec0,S2);
                fprintf('Solving using NFXP\n');
                ll_function   = @DDCMixture.ll_NFXP;
            else
                fprintf('No correct algorithm chosen, use EE\n');
                [vtil_1,p1_0] = EulerEquation.solve(S1.P,S1.state,S1.beta,theta_vec0);
                [v_til2,p2_0] = EulerEquation.solve(S1.P,S2.state,S2.beta,theta_vec0);
                ll_function   = @DDCMixture.ll_EE;
            end
            while (diff > opt.tol)
                ts = tic;
                %STEP 2: Then estimate the theta using the mixture weight
                f = @(theta_vec)(-ll_function(theta_vec,datasim,w0,p1_0,p2_0,S1,S2)); %Make sure the solver works
%                 f = @(theta_vec)(-EulerEquation.likelihood(theta_vec,datasim,p1_0,S1));
                Prob = conAssign(f,[],[],[],[],[],'2typeMixtureDDC',theta_vec0);
                Result = tomRun('Snopt',Prob);
                theta_vec = Result.x_k;
                
                if opt.output > 0
                    fprintf('Iteration: %d, Difference: %.4f, Time elapsed: %f Seconds \n',iter,diff,Result.REALtime);
                end                %STEP3: Update the weight using the theta
                %STEP 3: Update weight and q
                [ll,p1_1,p2_1,ll_1,ll_2] = ll_function(theta_vec,datasim,w0,p1_0,p2_0,S1,S2);
                
                minl = min(min(ll_1,ll_2));

                w1(:,1) = (q(1) * exp(ll_1 - minl) );
                w1(:,2) = (q(2) * exp(ll_2 - minl) );
                w1 = w1 ./sum(w1,2);
                diff = max(max(abs(w0 - w1)));
                w0 = w1;
                q = mean(w1); %The probability of each type
                % STEP 4: Update conditional choice probability
                theta_vec0 = theta_vec;
                [ll,p1_1,p2_1] = ll_function(theta_vec,datasim,w0,p1_0,p2_0,S1,S2);
                p1_0 = p1_1;
                p2_0 = p2_1;
                iter = iter + 1;

                
                if iter > opt.max_iter
                    break;
                end
            end
            w = w1;
            theta_hat = theta_vec;
        end %End of Mixture.sequential_EE
        
    end
end 