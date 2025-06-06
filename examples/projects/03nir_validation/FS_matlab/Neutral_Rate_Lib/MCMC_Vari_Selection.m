function [betam, sig2m, Gam, yfm, postmom_beta, postmom_sig2, postmom_Gam, postmom_yfm] = MCMC_Vari_Selection(Y, X, n0, n1, beta0, beta1, alpha0, delta0, IG0, IG1, x_f)

n = n0 + n1;
k = cols(X);

if nargin > 10
    Is_forecasting = 1; %  예측하기
else
    Is_forecasting = 0;  % 예측 안하기
end
%% 초기값 설정
if alpha0 > 2
    sig2 = delta0/alpha0;
else
    sig2 = stdc(Y)^2;
end
    
B0 = (IG0(2)/IG0(1))*eye(k);
B1 = (IG1(2)/IG1(1))*eye(k);

gam = ones(k, 1);

%% 저장할 방
Gam = zeros(n, k);
betam = zeros(n, k); 
sig2m = zeros(n, 1);
yfm = zeros(n, 1);

%% MCMC 시작
for iter = 1:n
        
    %  Step 1 : beta 샘플링
    beta = Gen_beta_VS(Y, X, sig2, gam, beta0, B0, beta1, B1);
    betam(iter, :) = beta';
    
    %  Step 2 : sig2 샘플링
    sig2 = Gen_Sigma(Y, X, beta, alpha0, delta0);
    sig2m(iter, :) = sig2;
    
    %  Step 3 : gam 샘플링
    gam = Gen_gam(beta, beta0, B0, beta1, B1);
    Gam(iter, :) = gam';
        
    % Step 4 : B0, B1 샘플링
    beta_0m = beta(gam == 0);
    b0 = Gen_Sigma_B(beta_0m, IG0(1), IG0(2));
    B0 = b0*eye(k);
    
    beta_1m = beta(gam == 1);
    b1 = Gen_Sigma_B(beta_1m, IG1(1), IG1(2));
    B1 = b1*eye(k);
    
    %  Step 5 : 예측하기
    if Is_forecasting == 1
       beta_gam = beta .* gam;
       yf = x_f'*beta_gam + sqrt(sig2)*randn(1, 1);
       yfm(iter, 1) = yf;
    end
end

betam = betam(n0+1:n, :); % 번인 버리기
sig2m = sig2m(n0+1:n, :); % 번인 버리기
Gam = Gam(n0+1:n, :); % 번인 버리기
yfm = yfm(n0+1:n, :);

alpha = 0.05;
maxac = 200;
postmom_beta = MHout(betam,alpha,maxac, 0);
postmom_sig2 = MHout(sig2m,alpha,maxac, 0);
postmom_Gam = MHout(Gam,alpha,maxac, 0);
postmom_yfm = MHout(yfm,alpha,maxac, 0);
end
%%
function beta = Gen_beta_VS(Y, X, sig2, gam, beta0, B0, beta1, B1)

B_ = diag((1 - gam')*diag(B0) + gam'*diag(B1));
beta_ = (1 - gam')*beta0 + gam'*beta1;
B_inv = inv(B_);
k = cols(X);

%% 레짐 1
sig2inv = 1/sig2;
B1inv = sig2inv*(X'*X) + B_inv;
B1 = inv(B1inv);
A = sig2inv*X'*Y + B_inv*beta_;
beta = B1*A + chol(B1)'*randn(k, 1);

    
end

%%
function gam = Gen_gam(beta, beta0, B0, beta1, B1)

k = rows(beta);
gam = zeros(k, 1);
for j = 1:k
    
    p0j = exp(lnpdfn(beta(j), beta0(j), B0(j,j)));
    p1j = exp(lnpdfn(beta(j), beta1(j), B1(j,j)));
    p1 = p1j/(p0j + p1j);
    gam(j) = rand(1,1) < p1;
end
   
end
%%
function sig2 = Gen_Sigma_B(Y, alpha0, delta0)

     %% 레짐 1
     ehat = Y;
     d1 = delta0 + ehat'*ehat;
     a1 = alpha0 + rows(Y);
     sig2 = randig(a1/2,d1/2,1,1);
     
     
end
%%
function sig2 = Gen_Sigma(Y, X, beta, alpha0, delta0)

     %% 레짐 1
     ehat = Y - X*beta;
     d1 = delta0 + ehat'*ehat;
     a1 = alpha0 + rows(X);
     sig2 = randig(a1/2,d1/2,1,1);
     
     
end