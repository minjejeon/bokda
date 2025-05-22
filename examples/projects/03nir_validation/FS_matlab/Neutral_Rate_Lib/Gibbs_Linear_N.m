function [betam, sig2m, postmom, R2] = Gibbs_Linear_N( Y, X, b_0, B_0, a_0, d_0, n0, n1, printi )
% Gibbs-Sampling with normal error

if nargin == 8
    printi = 1;
end

precB_0 = invpd(B_0); % Precision matrix of B_0
n = n0 + n1; % total number of iteration

k = cols(X);
betam = zeros(n, k); % beta를 저장할 공간
sig2m = zeros(n, 1); % sig2 저장공간

% 초기값 설정
if a_0*d_0 > 0
    sig2 = 0.5*d_0/(0.5*a_0 - 1); % initial value for 1/sig2
else
    sig2 = stdc(Y)^2;
end

for iter=1:n
    
    if printi == 1
        [~, resid] = minresid(iter,100);
        if resid == 0
            clc
            disp(['현재 반복시행은 ',num2str(iter)]);
        end
    end
    
    % Step1 : Full condtional posterior distribution of b, given sig2
    beta = Gen_beta(Y, X, b_0, precB_0, sig2);
    betam(iter,:) = beta'; % save beta
    
    % Step2: Full conditional posterior distribution of sig2, given b
    sig2 = Gen_sig2(Y, X, beta, a_0, d_0);
    sig2m(iter) = sig2;
    
end

% 번인을 버리기
betam = betam(n0+1:n, :); % burn-in period
sig2m = sig2m(n0+1:n, 1);

%% 결과보기
MHm = [betam, sig2m];
alpha = 0.05;
maxac = 200; % pacf, acf to 200 orders
postmom = MHout(MHm, alpha, maxac);

disp('================================================================');
disp('   Estimates    S.E.     2.5%      97.5%     Ineff ');
disp('----------------------------------------------------------------');
disp([postmom(:,2) postmom(:,3) postmom(:,4) postmom(:,6) postmom(:,7)]);
disp('=================================================================');

TSS = (Y-meanc(Y))'*(Y-meanc(Y));
beta_hat = inv(X'*X)*X'*Y;
E_hat = Y - X*beta_hat;
ESS = E_hat'*E_hat;
R2 = 1 - TSS/ESS;

T = rows(Y);
sig2hat = ESS/(T-k);
lnL = sumc(lnpdfn(E_hat, zeros(T, 1), sig2hat*ones(T,1)));
BIC = -2*lnL + k*log(T);
disp(['lnL is  ', num2str(lnL)]);
disp(['BIC is  ', num2str(BIC)]);
save MHm.txt -ascii MHm;
save postmom.txt -ascii postmom;

%% 사후분포 그림그리기
npara = cols(MHm); % 파라메터의 수
m1 = round(sqrt(npara));
m2 = ceil(sqrt(npara));
for i = 1:npara
    subplot(m1, m2, i);
    para = MHm(:, i);
    minp = minc(para); % 최소
    maxp = maxc(para); % 최대
    intvl = (maxp - minp)/50;
    interval = minp:intvl:maxp;
    histogram(para, interval);
    if i < npara
        xlabel(['beta ', num2str(i)])
    elseif i == npara
        xlabel('\sigma^2')
    end
end

end

function [ beta ] = Gen_beta(Y, X, b_0, precB_0, sig2)

k = cols(X);
XX = X'*X;
XY = X'*Y;

B_1 = invpd((1/sig2)*XX + precB_0); % full conditional variance B_1
A = (1/sig2)*XY + precB_0*b_0;
M = B_1*A;

beta = M + chol(B_1)'*randn(k, 1); % sampling beta


end

function [ sig2 ] = Gen_sig2( Y, X, beta, a_0, d_0 )

T = rows(X);
a_1 = T + a_0;
e = Y - X*beta;
d_1 = d_0 + e'*e;

sig2 = randig(a_1/2, d_1/2, 1, 1); % sig2 sampling


end


