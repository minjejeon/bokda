% Modified deterministic optimizer (DO) to include partial argument vector
% ** MORE IMPORTANTLY, the initial Hessian and its inverse are
% ** no longer needed. In case the Hessian is not p.d., a better
% ** approximation is calculated using cholmod within the program
% **
% ** INPUTS
% ** &f: pointer to function to be maximized
% ** &const: pointer to constraint procedure
% ** arg: full vector of argument
% ** index: index of argument components to be maximized over
% ** maxiter: max number of desired iterations
% ** co : = 1 if constrained optimizer, and 0 otherwise.
% ** printi: =1 to print intermediate results
% ** Note that in case of unconstrained optimization, 'const' is an arbitrary procedure. 
% **
% ** OUPUTS
% ** mu: maxima (full vector)
% ** fmax: maximized value of function
% ** g:  gradient at argmax
% ** V: Cov computed as inverse of negative Hessian (or a nearby p.d. matrix)
% ** Vinv: -Hessian
function [mu,fmax,g,V,Vinv] = DO_CKR2(FUN,CONSTR,arg,index,maxiter,co,printi,Sn)

argnew = arg;

if co == 1
    valid = CONSTR(argnew,Sn);  %/* check if parameter constraints are valid for initial value */

    if valid == 0
        disp('Starting values do not satisfy parameter constraints')
        disp('Setting initial likelihood to large negative value. Please check initial values.')
        disp(argnew)
    return
    end
   
end


db = 1;
iter = 1;
s = 1;

while db > 1e-6 && iter <= maxiter && s >= 1e-6 
    
    tic; % timer starts
       
    g = -Gradpnew1(FUN,argnew,index,Sn);
	H = -FHESSnew1(FUN,argnew,index,Sn);
    H = real(H);

	if rows(H) > 1
	
  		Hinv = invpd(H);

    	if isempty(Hinv) == 1
    	  	[Hc,indef,err] = cholmod(H);
    	  	H = Hc'*Hc;
       %     Hinv = invpd(H);
    	  	Hinvc = inv(Hc);
    	  	Hinv = Hinvc*Hinvc';
        end
    	
     else
		H = maxc([abs(H);1e-016]);
		Hinv = 1/H;
     end
    	
  	db = - Hinv*g'; 		% the change in estimate 
   	s = 1;
    s_hat = s;
   	x00 = argnew;
    fcd0 = FUN(x00,Sn);
    
   	x00(index) = x00(index) + s*db/2;
    
    %%%%%%% adjusting stepsize (s) until x00 satisfies the CONSTR  %%%%%%%
    if co == 1 && CONSTR(x00,Sn) == 0     % if constraint is not satisfied, reduce the stepsize */

    	while s > 0 && CONSTR(x00,Sn) == 0
    	    x00 = argnew;
         	x00(index) = x00(index) + s*db/2;
            
             	if s < 1e-6 
          	        x00 = argnew;
%          		    disp('close to constraint (minimum step reached)');
                    s = 0;
               end
             s_hat = s;   % output of this 'while' looping
             s = s/2;
      end
        
        fcd1 = FUN(x00,Sn);
        
        %%% Even if x00 satisfies the CONSTR, s_hat is not taken if the
        %%% value of function becomes smaller
        
        if fcd1 < fcd0
           x00 = argnew;
%            disp('close to constraint (minimum step reached)');
           s_hat = 0;
       end
    end
    
    %%%%%%% checking whether fcd1 > fcd0  %%%%%%%
%    	fcd0 = FUN(x00,Sn);  % current value of function
   	fcd1 = fcd0 - 1; % new value of function
    s = s_hat;
   	while fcd1 < fcd0
       
   		x00 = argnew;
   		x00(index) = x00(index) + s*db/2;

      	if co == 1 
      		valid = CONSTR(x00,Sn);
        else 
      		valid = 1;
        end
      	
      	if valid == 1 && s < 1e-6
% 	      	disp('Minimum step length reached. Result may be suboptimal');
			s = 0;
        end

        if  valid == 0
            fcd1 = fcd0 - 1;
        else
            fcd1 = FUN(x00,Sn); % new value of function
        end
        
        s_hat = s;
        s = s/2;
        
    end

    s = s_hat;
    argnew(index) = argnew(index) + s*db/2;
    
    iter_End = toc; % timer ends
    
    
    if printi == 1
        disp('===============================================================================');
        disp(['current DO iteration is ',num2str(iter)]);
	    disp(['current function value ',num2str(FUN(argnew,Sn))]);
        disp(['current step size ',num2str(s)]);
        disp('-------------------------------------');
        disp('     indices  argmax    gradient');
        disp('-------------------------------------');
        disp([index, argnew(index), g']);
        disp('-------------------------------------');
        TotalCT = maxiter*iter_End/3600;
        RemainingCT = (maxiter-iter)*iter_End/3600;
        disp( ['Computing time of each DO iteration in min is =  ', num2str(iter_End/60)]);
        disp( ['Total DO computing time in hours is =  ', num2str(TotalCT)]);
        disp( ['Remaining DO computing time in min is =  ', num2str(RemainingCT*60)]);
        disp( ['Remaining DO computing time in hours is =  ', num2str(RemainingCT)]);
        disp( ['Remaining DO computing time in days is =  ', num2str(RemainingCT/24)]);
        disp('===============================================================================');
    end

    db = maxc(abs(g'));
    iter = iter + 1;
  
end

mu = argnew; % maximum 
fmax = FUN(mu,Sn);  % value of function at maximum 
H = -FHESSnew1(FUN,mu,index,Sn);
g = Gradpnew1(FUN,mu,index,Sn);
H = real(H);
H = 0.5*(H + H');

    if rows(H) > 1
        [V, err] = invpd(H);
        if err == 1
            [Vinvc, ~] = cholmod(H);
            Vinv = Vinvc'*Vinvc;
            Vc = invuptr(Vinvc);
            V = Vc*Vc';
        else
            Vinv = H;
        end
    
    else  % in case H is a scalar
        Vinv = maxc([abs(H);1e-016]);
        V = 1/Vinv;
    end
end

     