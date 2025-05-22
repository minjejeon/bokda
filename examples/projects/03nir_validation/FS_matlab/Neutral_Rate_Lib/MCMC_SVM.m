function [Volm,Hm, MHm, Yfm, PredLikm] = MCMC_SVM(ym, yf, n0, n1, mu_phi_, precb_, v_, d_, isForecast)

n = n0 + n1;
% 초기값 설정하기
mu_phi = mu_phi_;
sig2 = d_/v_;
sig2_inv = 1/sig2;

ysm = log(ym.*ym); %선형화를 위한 자료변환
hm = ysm;
T = rows(ysm);

% normal mixture를 이용한 log(Chi-square(1))의 approximation
pm = zeros(7,1);
pm(1) = 0.00730;
pm(2) = 0.10556;
pm(3) = 0.00002;
pm(4) = 0.04395;
pm(5) = 0.34001;
pm(6) = 0.24566;
pm(7) = 0.25750;

msm = zeros(7,1);
msm(1) = -11.40039;
msm(2) = -5.24321;
msm(3) = -9.83726;
msm(4) = 1.50746;
msm(5) = -0.65098;
msm(6) = 0.52478;
msm(7) = -2.35859;

vsm = zeros(7,1);
vsm(1) = 5.79596;
vsm(2) = 2.61369;
vsm(3) = 5.17950;
vsm(4) = 0.16735;
vsm(5) = 0.64009;
vsm(6) = 0.34023;
vsm(7) = 1.26261;


% MCMC output을 저장할 방
Hm =  zeros(n,T);
Sig2m = zeros(n,1);
mu_phim = zeros(n,2);
Volm =  zeros(n,T);
Yfm = zeros(n,1);
PredLikm = zeros(n,1);
freq = 20;
for iter = 1:n
    
    % report intermediate results
    if isequal(floor(iter/freq),iter/freq) == 1  % 1 if equal
        clc
        disp(['현재 반복시행은 ', num2str(iter)]);
    end
    
    %  Step 1 : mu와 phi sammpling
    [mu_phi] = Gen_mu_phi(hm,mu_phi_,precb_,sig2_inv,mu_phi);
    mu_phim(iter,:) = mu_phi'; % mu와 phi 저장하기
    
    %  Step 2 : ht의 조건부 분산
    [sig2_inv, sig2] = Gen_Sigma(hm,v_,d_,mu_phi);
    Sig2m(iter,1) = sig2;
    
    %  Step 3 : Sampling sm
    sm = Gen_Sm(ysm,hm,pm,msm,vsm);
    
    %  Step 4 : Sampling hm
    hm = Gen_Fm(ysm,mu_phi,sig2,sm,msm,vsm);
    Hm(iter,:) = hm(:, 1)';
    
    vol = exp(hm/2); % volatility
    Volm(iter,:) = vol(:, 1)'; % volatility 저정하기
    
    [y_pred, lnpredlik] = Gen_Forecast(mu_phi, sig2, ym, hm, yf, isForecast);
    
    Yfm(iter) = y_pred;
    PredLikm(iter) = exp(lnpredlik);
    
end

%% Summary of Output
MHm = [mu_phim Sig2m];
MHm = MHm(n0+1:n,:);

% 사후분포 그리기
Volm = Volm(n0+1:n,:);
Hm = Hm(n0+1:n,:);
Yfm = Yfm(n0+1:n, :);   % 번인 버리기
if isForecast == 1
    PredLikm = PredLikm(n0+1:n, :);   % 번인 버리기
end


alpha = 0.025;
maxac = 200;
postmom = MHout(MHm,alpha,maxac);

disp('=========================================================');
disp('  MCMC results');
disp('=========================================================');
disp('   Estimates  S.D.  ineff');
disp('---------------------------------------------------------');
disp([postmom(:,2) postmom(:,3) postmom(:,7)]);
disp('---------------------------------------------------------');

if isForecast == 1
    predlik = meanc(PredLikm);
    disp(['Predictive likelihood is  ', num2str(predlik)]);
    save Yfm.txt -ascii Yfm;
    save yf.txt -ascii yf;
    
    %% Expected Shortfall 계산하기
    VaR = quantile(Yfm, 0.05);
    disp(['VaR is  ', num2str(VaR), ' %']);
    ES_Yfm = Yfm(Yfm<VaR);
    ES = meanc(ES_Yfm);
    disp(['ES is  ', num2str(ES), ' %']);
    ql = [0.025 0.975];
    cb = quantile(Yfm, ql);
    disp(['95% C.I. is  [', num2str(cb),']']);
end

volm = meanc(Volm);

save volm.txt -ascii volm;


%% 그림 그리기
T = rows(ym);
xa = 1:T;

if isForecast > 0
    subplot(2,1,1)
    plot(xa, abs(ym), 'k:', xa, volm, 'b-');
    title('(a) Volatility')
    xlim([0 T+1])
    xlabel('Time')
    legend('|y|','volatility');
    
    subplot(2,1,2)
    miny = minc(Yfm);
    maxy = maxc(Yfm);
    int = (maxy - miny)/50;
    interval = miny:int:maxy;
    hist(Yfm, interval);
    title('(b) Predictive Dist.')
else
    plot([abs(ym), volm]);
    title('Volatility')
    legend('|y|','변동성 추정치');
end

end

%% State 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function sm = Gen_Sm(ysm,hm,pm,msm,vsm)

T = rows(ysm);
sm = zeros(T,1);

for t = 1:T
    
    fm = zeros(7,1);
    for i = 1:rows(pm)
        y_tl = msm(i) + hm(t);  % ystm(t)의 조건부 기대값
        ft = lnpdfn(ysm(t),y_tl,vsm(i)) + log(pm(i));
        fm(i) = exp(ft);
    end
    
    prb_st = fm/(sumc(fm)); % st의 사후 확률
    sm(t) = discret1(prb_st,1);
    
end

end



%% Sigma 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [sig2_inv,sig2] = Gen_Sigma(hm,v_,d_,mu_phi)

hm = hm(:, 1);
Y = hm(2:end);
X = [ones(rows(hm)-1,1) hm(1:end-1)];
T = rows(Y);

ehat = Y - X*mu_phi; % 잔차항
v1 = v_ + T;
d1 = d_ + ehat'*ehat;
sig2 = randig(v1/2,d1/2,1,1);
sig2 = sig2(1,1);
sig2_inv = 1/sig2;


end

%% mu와 phi 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [mu_phi] = Gen_mu_phi(hm,b_,precb_,sig2_inv,mu_phi0)

hm = hm(:, 1);
Y = hm(2:end);
X = [ones(rows(hm)-1,1) hm(1:end-1)];
k = cols(X);
XX = X'*X;
XY = X'*Y;

varb1 = invpd(precb_ + sig2_inv*XX);  % full conditional variance
b1 = varb1*(precb_*b_ + sig2_inv*XY); % full conditional mean

mu_phi = b1 + chol(varb1)'*randn(k,1); % beta sampling 하기

if abs(mu_phi(2)) > 1  % phi의 절대값이 1보다 작도록 제약
    mu_phi = mu_phi0;
end

end


%% 예측분포 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [y_pred, lnpredlik] = Gen_Forecast(mu_phi, sig2, ym, hm, yf, isForecast)

T = rows(ym);

mu = mu_phi(1);
phi = mu_phi(2);

hL = hm(T, 1); % 예측오차
hf = mu + phi*hL + sqrt(sig2)*randn(1,1);

volf = exp(hf/2);
y_pred = sqrt(volf)*randn(1,1);

if isForecast == 1
    lnpredlik = lnpdfn(yf, 0, volf);
else
    lnpredlik = 0;
end

end


%% Fm 샘플랭 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Fm = Gen_Fm(ysm,mu_phi,Omega,sm,msm,vsm)

T = rows(ysm);
mu = mu_phi(1);
C = mu;
phi = mu_phi(2);
G = phi;
Q = Omega;

%%%%% Kalman filtering step
f_ttm = zeros(T,1);
P_ttm = zeros(T,1);
f_ll = C/(1-G);
P_ll = Q/(1-G^2);

for t = 1:T
    
    st = sm(t);
    f_tl = C + G*f_ll;
    P_tl = P_ll*G^2 + Q;
    var_tl = P_tl +  vsm(st);
    var_tlinv = 1/var_tl;
    
    e_tl = ysm(t) - msm(st) - f_tl;
    Kalgain = P_tl*var_tlinv;
    f_tt = f_tl + Kalgain*e_tl;
    P_tt = 1 - Kalgain;
    P_tt = P_tt*P_tl;
    
    f_ttm(t,1) = f_tt;
    P_ttm(t,1) = P_tt;
    
    f_ll = f_tt;
    P_ll = P_tt;
    
end

%%% Backward recursion
Fm = zeros(T,1);  % T by km

P_tt = P_ttm(T,1);  % km by km
cP_tt = sqrt(P_tt); % km by km

f_tt = f_ttm(T,1); % km by 1
ft = f_tt + cP_tt'*randn(1,1);  % km by 1
Fm(T,1) = ft'; % 1 by by km
t = T - 1; %  time index

while t >= 1
    
    f_tt = f_ttm(t,1);  % km3 by 1
    P_tt = P_ttm(t,1);  % km3 by km3
    
    GPG_Q = P_tt*G^2 + Q; % km by km
    GPG_Qinv = 1/GPG_Q; % km by km
    
    e_tl = Fm(t+1,1) - C - G*f_tt; % km by 1
    
    PGG = P_tt*G*GPG_Qinv;  % km3 by km
    f_tt1 = f_tt + PGG*e_tl;  % km3 by 1
    
    PGP = PGG*G*P_tt;  % km3 by km3
    P_tt1 = P_tt - PGP;
    
    cP_tt1 = sqrt(P_tt1);
    
    ft = f_tt1 + cP_tt1*randn(1,1);
    Fm(t,1) = ft; % 1 by by km
    
    t = t - 1;
end


end
