function [MHm, F1m, F2m, F3m, Yfm, postmom] = MCMC_DNS(n0, n1, Spec)
% MHm = 파라메터의 사후 분포
% F1m = 수준 요인의 사후 분포
% F2m = 기울기 요인의 사후 분포
% F3m = 곡률 요인의 사후 분포
% Yfm = 사후 예측 분포

Ym = Spec.Ym;
isforecast = Spec.isforecast;

if isforecast == 1 % 예측하고자 할 때
    realized_yc = Ym(end,:); % 실현된 out-of-sample 수익률곡선
    ym = Ym(1:end-1,:); % in-sample
else  % 예측 안 할때
    ym = Ym;
end

[T, N] = size(ym); % N = 만기의 수

n = n0 + n1;

k = Spec.k; % factor의 수
kki = k*2;
nb1 = kki;   % mu and G
nb2 = k^2;   % Omega
nb3 = N;   % Sigma
nb = [nb1;nb2;nb3];

nmh = sumc(nb); % 파라메터 수

upp = cumsum(nb);
low = [0;upp(1:length(nb)-1)] + 1;

%% 파라메터의 인덱스
indv = 1:nmh;
indv = indv';
ind_muG = indv(low(1):upp(1));
ind_Omega = indv(low(2):upp(2));
ind_Sig = indv(low(3):upp(3));

Spec.ind_muG = ind_muG;
Spec.ind_Omega = ind_Omega;
Spec.ind_Sig = ind_Sig;

% Factorloading
tau = Spec.tau;
Gamma = makeGam(tau,Spec.lambda);
Spec.Gamma = Gamma;

% 초기값 설정하기
theta = zeros(nmh,1);
Omega = eye(k);
Omega_inv = invpd(Omega);
Fm = GenFm1(ym, tau, Spec.lambda); % factor의 초기값
theta(ind_muG) = Spec.b_;
theta(ind_Omega) = vec(Omega);
theta(ind_Sig) = 1*ones(N,1);

%% 저장할 방
Yfm = zeros(n, N); % one-month-ahead 예측분포 저장할 방
MHm = zeros(n,nmh);
F1m = zeros(n,T);
F2m = zeros(n,T);
F3m = zeros(n,T);

%% MCMC
for iter = 1:n
    
    %  Step 1: Posterior Conditional Distribution of b, given omega
    [theta, gam, Ftm, FLm] = Gen_mu_G(Fm,theta, Omega_inv, Spec);
    
    %  Step 2: Posterior Conditional Distribution of omega, given b
    [theta, Omega, Omega_inv] = Gen_Omega(Ftm,FLm,theta, Spec);
    
    %  Step 3: Posterior Conditional Distribution of sigma, given omega and b
    [theta, diag_Sigma] = Gen_Sigma(ym,Fm,theta,Spec);
    MHm(iter,:) = theta';
    
    %  Step 4 : Sampling Fm
    Fm = Gen_Fm(ym,theta, Omega, diag_Sigma, Spec);
    F1m(iter,:) = Fm(:,1)';
    F2m(iter,:) = Fm(:,2)';
    F3m(iter,:) = Fm(:,3)';
    
    % 예측
    if isforecast == 1
        [Yfm] = Forecast(theta,Omega,diag_Sigma,Fm,iter,Yfm, Spec);
    end
    
    % 중간결과보기
    if isequal(floor(iter/20),iter/20) == 1  % 1 if equal
        clc
        prt2(gam,Omega,diag_Sigma,iter);
    end
    
end



%% burn-in 버리기
Yfm = Yfm(n0+1:n,:);
MHm = MHm(n0+1:n,:);
F1m = F1m(n0+1:n,:);
F2m = F2m(n0+1:n,:);
F3m = F3m(n0+1:n,:);

mF1m = meanc(F1m);
mF2m = meanc(F2m);
mF3m = meanc(F3m);

mFm = [mF1m mF2m mF3m];
%%  결과보기
alpha = 0.025;
maxac = 200;
postmom = MHout(MHm,alpha,maxac);

disp('==================================================');
disp('     index   Mean     S.E.      2.5%      97.5%     Ineff.');
disp([postmom(:,1:4) postmom(:,6) postmom(:,7)]);
disp('==================================================');
save MHm.txt -ascii MHm;
save mFm.txt -ascii mFm;
save Yfm.txt -ascii Yfm;
save realized_yc.txt -ascii realized_yc;

%% Factor 그리기
xtick = 13:36:T;
xa = 1:T;
xticklabel = {'2002:M1', '2005:M1', '2008:M1', '2011:M1'};
subplot(2,1,1)
h = plot(xa, mFm(:,1), 'b-',xa, mFm(:,2), 'k--',xa, mFm(:,3), 'k:' );
xlim([0, T+1]);
xlabel('Time');
set(gca,'XTick', xtick)
set(gca,'XTickLabel',xticklabel,'fontsize', 10)
set(h,'Linewidth',2)
legend('Level','Slope','Curvature')
title('(a) Factors');

%% 예측분포 그리기
q = [0.05, 0.5, 0.95];
pred_yc = quantile(Yfm, q)';
subplot(2,1,2)
h = plot(tau, pred_yc(:,1), 'k:',tau, pred_yc(:,2), 'k--',tau, pred_yc(:,3), 'k:',tau, realized_yc, 'b-');
xlabel('Maturity');
xlim([tau(1), tau(end)]);
ylim([minc1(Yfm) maxc1(Yfm)]);
set(gca,'XTick', tau)
set(h,'Linewidth',2)
legend('5%', 'Median', '95%', 'Realized','location','Southeast')
title('(b) Predictive yield curve');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Omega 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [theta,Omega,Omega_inv] = Gen_Omega(Y,X, theta, Spec)

R0 = Spec.R0;
nu = Spec.nu;
ind_muG = Spec.ind_muG;
ind_Omega = Spec.ind_Omega;

beta = theta(ind_muG);
T = rows(Y);
ehat2m = 0; % 잔차항의 제곱의 합 
for t = 1:T
    Xt = X(:,:,t);
    ehat = Y(t,:)' - Xt*beta; % 잔차항
    ehat2m = ehat2m + ehat*ehat'; % p by p
end

Omega1_inv = ehat2m + invpd(R0);
Omega1 = invpd(Omega1_inv);
Omega_inv = randwishart(Omega1,(T+nu));
Omega = invpd(Omega_inv);

theta(ind_Omega) = vec(Omega);  % Omega 저장하기
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 예측분포 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Yfm] = Forecast(theta, Omega,diag_Sigma,Fm,iter,Yfm, Spec)   

ind_muG = Spec.ind_muG;
Gamma = Spec.Gamma;
k = cols(Gamma);

mu = makeMu(theta, ind_muG);
G = makeG(theta, ind_muG);

T = rows(Fm);
N = rows(Gamma);
f_L = Fm(T,:)';
chol_omg = cholmod(Omega)';

  
    factor_shock = chol_omg*randn(k,1);
    f_t = mu + G*f_L + factor_shock;  % k by 1
    
    measurement_err = sqrt(diag_Sigma).*randn(N,1);
    y_f = Gamma*f_t + measurement_err;  % N by 1
    
    Yfm(iter,:) = y_f';


    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Sigma 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [theta, diag_Sigma] = Gen_Sigma(ym,Fm,theta,Spec)
ind_Sig = Spec.ind_Sig;
a0 = Spec.a0;
d0 = Spec.d0;
Gamma = Spec.Gamma;

     T = rows(ym);
     N = cols(ym);
     
     Fitted = Fm*Gamma';
     ehatm = ym - Fitted; % 잔차항
     
     v1 = a0 + T;
     diag_Sigma = zeros(N,1);
     for indtau = 1:N
         ehat = ehatm(:,indtau); % 잔차항
         d1 = d0 + 5000*(ehat'*ehat);
         diag_Sigma(indtau) = randig(v1/2,d1/2,1,1)/5000;         
     end
 
    theta(ind_Sig) = diag_Sigma;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 중간결과보고 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  prt2(beta,Omega,diag_Sigma,iter)

mu = [beta(1);beta(3);beta(5)];
g = [beta(2);beta(4);beta(6)];
Vol = diag(Omega);
Rho = corrcov(Omega);

disp('==================================');
disp( [' current iteration is ', num2str(iter)]); 
disp('----------------------------------');
disp( ['mu =  ', num2str(mu')]);  
disp( ['G =  ', num2str(g')]);
disp('----------------------------------');
disp( ['Factor 변동성 =  ', num2str(Vol')]);
disp( ['Recaled Sigma =  ', num2str(5000*diag_Sigma')]);
disp('----------------------------------');
disp( 'Factor 상관관계 ');
disp(Rho);
disp('----------------------------------');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% mu와 G 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [theta,beta,Ftm,FLm] = Gen_mu_G(Fm, theta, Omega_inv, Spec)

b_ = Spec.b_;
var_ = Spec.var_;
ind_muG = Spec.ind_muG;

T = rows(Fm);
k = cols(Fm);

Ftm = Fm(2:T,:); % 종속변수
Y = Ftm;

% 설명변수 만들기
F1L = [ones(T-1,1) Fm(1:T-1,1)];
F2L = [ones(T-1,1) Fm(1:T-1,2)];
F3L = [ones(T-1,1) Fm(1:T-1,3)];

FL = [F1L F2L F3L];

ki = cols(F1L); % 각 식에 있는 설명변수의 수
kki = k*ki;

FLm = zeros(k,kki,T-1); % 설명변수를 새롭게 저장할 방

for t = 1:(T-1)
    xt = zeros(k,kki);
    for indp = 1:k
        
        xt(indp,(indp-1)*ki+1:indp*ki) = FL(t,(indp-1)*ki+1:indp*ki);
        
    end
    
    FLm(:,:,t) = xt; % p by k
end

X = FLm; % 설명변수, 3차원

    XX = 0;
    XY = 0;
    for t = 1:(T-1)
        Xt = X(:,:,t);
        XX = XX + Xt'*Omega_inv*Xt;
        XY = XY + Xt'*Omega_inv*Y(t,:)';
    end

    precb_ = invpd(var_); 
    Bn_inv = precb_ + XX;
    Bn_inv = 0.5*(Bn_inv + Bn_inv');
    varb1 = invpd(Bn_inv);
    varb1 = 0.5*(varb1 + varb1');
    b1 = varb1*(precb_*b_ + XY); % full conditional mean  
    Chol_varb1 = chol(varb1)';
    beta = b1 + Chol_varb1*randn(kki,1); % beta sampling 하기
    theta1 = theta;
    theta1(ind_muG) = beta;
    
    valid = paramconst(theta1, Spec); % 안정성 제약 만족여부 확인
    if valid == 1
        theta = theta1;
    else
        beta = theta(ind_muG);
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Factor 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Fm] = Gen_Fm(ym,theta, Omega, diag_Sigma, Spec)

T = rows(ym);
ind_muG = Spec.ind_muG;
Gamma = Spec.Gamma;
k = cols(Gamma);

mu = makeMu(theta, ind_muG);
G = makeG(theta, ind_muG);

%%%%% Kalman filtering step 
f_ttm = zeros(k,1,T);
P_ttm = zeros(k,k,T);
f_ll = (eye(k) - G)\mu;
P_ll = makeR0(G,Omega); % 비조건부 분산-공분산 행렬
Q = Omega;
Sigma = diag(diag_Sigma);

for t = 1:T
   
   f_tl = mu + G*f_ll;
   P_tl = G*P_ll*G' + Q;
   var_tl = Gamma*P_tl*Gamma' +  Sigma;
   var_tl = 0.5*(var_tl + var_tl');
   var_tlinv = invpd(var_tl);
   
   e_tl = ym(t,:)' - Gamma*f_tl;
   Kalgain = P_tl*Gamma'*var_tlinv;
   f_tt = f_tl + Kalgain*e_tl;
   P_tt = eye(k) - Kalgain*Gamma;
   P_tt = P_tt*P_tl;
   
   f_ttm(:,:,t) = f_tt;
   P_ttm(:,:,t) = P_tt;
   
   f_ll = f_tt;
   P_ll = P_tt;
   
end

%%% Backward recursion
Fm = zeros(T,k);  % T by k

P_tt = P_ttm(:,:,T);  % k by k
P_tt = (P_tt + P_tt')/2;
cP_tt = cholmod(P_tt); % k by k

f_tt = f_ttm(:,:,T); % k by 1
ft = f_tt + cP_tt'*randn(k,1);  % k by 1
Fm(T,:) = ft'; % 1 by by k
t = T - 1; %  time index

for t = (T-1):-1:1
    
  f_tt = f_ttm(:,:,t);  % km3 by 1
  P_tt = P_ttm(:,:,t);  % km3 by km3
  
  GPG_Q = G*P_tt*G' + Q; % k by k
  GPG_Q = (GPG_Q + GPG_Q')/2;
  GPG_Qinv = invpd(GPG_Q); % k by k
    
  e_tl = Fm(t+1,:)' - mu - G*f_tt; % k by 1
  
  PGG = P_tt*G'*GPG_Qinv;  % km3 by k
  f_tt1 = f_tt + PGG*e_tl;  % km3 by 1
 
  PGP = PGG*G*P_tt;  % km3 by km3
  P_tt1 = P_tt - PGP;
  
  P_tt1 = (P_tt1 + P_tt1')/2;
  cP_tt1 = cholmod(P_tt1);

  ft = f_tt1 + cP_tt1'*randn(k,1);
  Fm(t,:) = ft'; % 1 by by k
  
  t = t - 1;
end

end

%% Factor의 초기값 계산하기
function [Fm] = GenFm1(data, tau, lambda)

indbasis = [1;3;rows(tau)]; % basis yields

ym_basis = data(:,indbasis);
Gamma = makeGam(tau,lambda);

n = rows(ym_basis);
k = cols(Gamma);
bbar_Binv = inv(Gamma(indbasis,:));
Fm = zeros(n,k);

for t = 1:n
  ft = bbar_Binv*ym_basis(t,:)'; % k by 1
  Fm(t,:) = ft'; % 1 by k
end

end

%% G 만들기
function G = makeG(psi, ind_muG)

beta = psi(ind_muG);
G = [beta(2);beta(4);beta(6)];
G = diag(G);

end

%% factorloading 만들기
function Gamma = makeGam(tau,lambda)
  
    ntau = rows(tau);
    Gamma = ones(ntau, 3);
        
    Gamma(:,2) = (ones(ntau,1)-exp(-tau*lambda))./(tau*lambda);
    Gamma(:,3) = Gamma(:,2) - exp(-tau*lambda);

end

%% mu 만들기
function mu = makeMu(psi, ind_muG)

beta = psi(ind_muG);
mu = [beta(1);beta(3);beta(5)];

end

%% Macro to compute unconditional variance of the factors
function R0 = makeR0(G,omega)
 k = rows(G);
 k2 = k^2;
 G2 = kron(G,G);
 eyeG2 = eye(k2)-G2;
 omegavec = reshape(omega,k2,1);
 R00 = (eyeG2)\omegavec; %  vec(R00)
 R0 = reshape(R00,k,k)';
 R0 = (R0 + R0')/2;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 제약 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [valid] = paramconst(theta, Spec)

  validm = ones(30,1);
  
  if minc(isfinite(theta)) == 0;
     validm(30) = 0;
  end
  
  if maxc(isnan(theta)) == 1;
     validm(29) = 0;
  end
  ind_muG = Spec.ind_muG;
  G = makeG(theta, ind_muG);
  validm(1) = maxc(abs(diag(G))) < 1;
  
  valid = minc(validm); % if any element is equal to zero, invalid
  
  end
