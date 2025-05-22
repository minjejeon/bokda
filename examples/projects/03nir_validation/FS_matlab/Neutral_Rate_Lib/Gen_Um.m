%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Factor 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Um, Uttm] = Gen_Um(Ym, Um, rho, Phi, Sigma, Omega)

[T, M] = size(Ym);

H = [eye(M), eye(M)];
H(4, 1:2) = rho';
k = cols(H);

Q = eye(2*M);
Q(1:M, 1:M) = Omega;
Q(M+1:2*M, M+1:2*M) = Sigma;

F = eye(2*M);
F(M+1:2*M, M+1:2*M) = Phi;

%%%%% Kalman filtering step 
f_ttm = zeros(k, T);
P_ttm = zeros(k, k,T);
f_ll = Um(1, :)';
P_ll =10*eye(k); % 비조건부 분산-공분산 행렬
P_ll(1,1) = 0.1;
P_ll(M+1,M+1) = 0.1;

for t = 1:T
   f_tl = F*f_ll;
   P_tl = F*P_ll*F' + Q;
   var_tl = H*P_tl*H';
   var_tl = 0.5*(var_tl + var_tl');
   var_tlinv = invpd(var_tl);
   
   e_tl = Ym(t,:)' - H*f_tl;
   Kalgain = P_tl*H'*var_tlinv;
   f_tt = f_tl + Kalgain*e_tl;
   P_tt = eye(k) - Kalgain*H;
   P_tt = P_tt*P_tl;
   
   f_ttm(:, t) = f_tt;
   P_ttm(:,:,t) = P_tt;
   
   f_ll = f_tt;
   P_ll = P_tt;
   
end

%%% Backward recursion
Fm = zeros(T, k);  % T by k

P_tt = P_ttm(:,:,T);  % k by k
P_tt = (P_tt + P_tt')/2;
cP_tt = cholmod(P_tt); % k by k

f_tt = f_ttm(:, T); % k by 1
ft = f_tt + cP_tt'*randn(k, 1);  % k by 1
Fm(T,:) = ft'; % 1 by by k

for t = (T-1):-1:1
    
  f_tt = f_ttm(:, t);  % km3 by 1
  P_tt = P_ttm(:,:,t);  % km3 by km3
  
  GPG_Q = F*P_tt*F' + Q; % k by k
  GPG_Q = (GPG_Q + GPG_Q')/2;
  GPG_Qinv = invpd(GPG_Q); % k by k
    
  e_tl = Fm(t+1,:)' - F*f_tt; % k by 1
  
  PGG = P_tt*F'*GPG_Qinv;  % km3 by k
  f_tt1 = f_tt + PGG*e_tl;  % km3 by 1
 
  PGP = PGG*F*P_tt;  % km3 by km3
  P_tt1 = P_tt - PGP;
  
  P_tt1 = (P_tt1 + P_tt1')/2;
  cP_tt1 = cholmod(P_tt1);

  ft = f_tt1 + cP_tt1'*randn(k,1);
  Fm(t,:) = ft'; % 1 by by k
  
end

Um = Fm;
Uttm = f_ttm';

end