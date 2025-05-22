function  NIR_BOK(Spec)

interval = 10;
MCMC_size = 500;
n1 = interval*MCMC_size;    % 시뮬레이션 크기
n0 = 1000;    % 번인
n = n0 + n1;
smoothing = 1600;

%%
% Y_Names_KOR = Spec.Y_Names_KOR;
Ym = Spec.Ym;
% Wm_raw = Spec.Wm_raw;
date = Spec.date;

cut = 0;
date = date(cut+1:end, 1);
Spec.date = date;
Ym = Ym(cut+1:end, :);

[G, ghat] = hpfilterK(Ym(:, 1), smoothing); %1600

[P, phat] = hpfilterK(Ym(:, 2), smoothing);

[C, chat] = hpfilterK(Ym(:, 3), smoothing);

[Z, ihat] = hpfilterK(Ym(:, 4) - P - G, smoothing);

G(1:4) = meanc(Ym(1:4, 1));
ghat(1:4) =  Ym(1:4, 1) - G(1:4);

Trend = [G, P, C, Z];
Cycle = [ghat, phat, chat, ihat];

[T, M] = size(Trend);
omega_ = eye(M);
ehat = Trend(2:end, 1) - Trend(1:end-1, 1);
omega_(1, 1) = 0.75*(ehat'*ehat)/T; % 잠재성장률
  
ehat = Trend(2:end, 2) - Trend(1:end-1, 2);
omega_(2, 2) = 0.5*(ehat'*ehat)/T; % 물가상승률

ehat = Trend(2:end, 3) - Trend(1:end-1, 3);
omega_(3, 3) = ehat'*ehat/T; % 가계부채

ehat = Trend(2:end, 4) - Trend(1:end-1, 4);
omega_(4, 4) = 3*(ehat'*ehat)/T; % 기타요인

[~, Sigma_hat] = OLS_VAR(Cycle, 1);
Sigma = diag(diag(Sigma_hat));
Sigma_ = Sigma;
Sigma_inv = invpd(Sigma);

Um = [Trend, Cycle];
Um_in = Um;
%%
[T, M] = size(Ym);
K = 2*M; % factor의 수


%% 사전분포
Phi_ = 0.0*eye(M);
b_= vec(Phi_);
var_ = 0.04*ones(M, M);
if Spec.No_FinCycle == 1
   var_(1:2, 3) = 0.0000000001;
   var_(4, 3) = 0.0000000001;
end
var_ = diag(vec(var_));


rho_ = [1; 1];
V_rho = diag([0.0001; 0.001]);

nu = round(T);
R0 = inv(Sigma_)/nu;  % Note that E(inv(Omega)) = nu*R0, Omega의 샘플링에 사용

% Omega
a_0 = round(T)*ones(M, 1);
d_0 = diag(omega_).*a_0;

%% 저장할 방과 초기값
Phi = 0.5*eye(M);
Omega = omega_;

rhom = zeros(n, 2);
Post_Um = zeros(n1, T, 2*M);
Rstm = zeros(n1, T);
Phim = zeros(n1, M^2);
Eyestm = Rstm;
for iter = 1:n
    
     counter_iter(iter, 1000, n);
     
    %% rho sampling23
    rho = Gen_rho(Ym, Trend, Cycle, rho_, V_rho, Omega);
    rhom(iter, :) = rho';
    
    %% Phi , Sigma sampling
    [Phi, Sigma, Sigma_inv] = Gen_Phi_Sigma(Phi, Cycle, b_, var_, Sigma_inv,nu,R0);

    %% Omega sampling
    Omega = Gen_Omega(Trend, a_0, d_0);
 
    %% 추세와 순환 샘플링
    [Um, Uttm] = Gen_Um_mex(Ym, Um_in, rho, Phi, Sigma, Omega); 
    
     Trend = Um(:, 1:M);
     Cycle = Um(:, M+1:2*M);
     
    if iter > n0
      
       for k = 1:K
              Post_Um(iter-n0, :, k) = Um(:, k)';
       end
            
     G = Trend(:, 1);
     P = Trend(:, 2);
     Z = Trend(:, 4);
     Rstm(iter-n0, :) = rho(1,1)*G' + Z';
     Eyestm(iter-n0, :) = Rstm(iter-n0, :) + rho(2,1)*P';
     Phim(iter-n0, :) = vec(Phi)';
    end
    
end

save Post_Um Post_Um
save rhom rhom
save Spec Spec
save Rstm Rstm
save Eyestm Eyestm
save Phim Phim
end