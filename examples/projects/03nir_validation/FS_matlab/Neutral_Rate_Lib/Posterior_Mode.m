clear
clc
load path1 
load path2 

addpath(path1)
addpath(path2)
%%
load Spec
load Data 
load Mum 
load Gamma_save 
load Sigma_save 
load Omega_save 
load beta_save 
load q_save 

%%%%%%%%%%%%%%%%%%%%
%% 1단계: 사후모드 계산하기
%%%%%%%%%%%%%%%%%%%%
is_Nonlinear = Spec.is_Nonlinear;
n1 = Spec.n1;
n0 = Spec.n0;
P = Spec.P;

% 사전밀도 초모수
beta_0 = Spec.beta_0;
B_0 = Spec.B_0;
diagB_0 = diag(B_0);
NonZeroRestriction = 1000*diagB_0 > 0.01;  % beta 중에 영(zero) 제약이 반영된 위치
Spec.NonZeroRestriction = NonZeroRestriction;
c_00 = Spec.c_00;
c_10 = Spec.c_10;
a_00 = Spec.a_00;
a_10 = Spec.a_10;
a_q0 = Spec.a_q0;
c_q0 = Spec.c_q0;
nu = Spec.nu;
Omega0 = Spec.Omega0;
Psi0 = Omega0*nu;  

Ym = demeanc(Data);
M = cols(Ym);
[Y0, YLm] = makeYX_mex(Ym, P);

lnLm = zeros(n1, 1); % 우도를 저장할 방
lnPriorm = zeros(n1, 1); % 사전밀도를 저장할 방
lnPost_st = -exp(20);
load time_Trend
for iter=1:n1
    
    if rem(iter, 100) == 0
        clc
        disp('Computing lnlik over MCMC cycles') 
        disp(['current iter is ', num2str(iter)])
        disp(['lnLst = ',  num2str(lnLst)]);
        disp(['lnPiorst = ',  num2str(lnPiorst)]);
        disp(['lnPostst = ',  num2str(lnPost_st)]);
    end
    
    %% 파라메터 가져오기
        beta = beta_save(iter, :)';
        Phi = reshape(beta, P*M, M)';  % p*k by k
        Omega = reshape(Omega_save(iter, :)', M, M);
        diag_Sigma = Sigma_save(iter, :)';
        q = q_save(iter);
        gamma = Gamma_save(iter, :)';
        Mu = Mum(:, :, n0+iter);
        
    %% 우도함수 계산하기
%         lnL = lnlik_mex(Mu, Y0, YLm, beta, Phi, Omega, diag_Sigma, gamma, is_Nonlinear);
        Y_tilm = Ym - Mu;
        [Y0_tilde, YLm_tilde] = makeYX_mex(Y_tilm, P); % VAR(P) 모형의 종속변수와 설명변수 생성
        lnL = lnlik_Linear_mex(Y0_tilde, YLm_tilde, beta, Omega);
        lnLm(iter) = lnL;
           
       
    %% 사전밀도 계산하기
    lnprior_beta = sumc(lnpdfn(beta(NonZeroRestriction==1), beta_0(NonZeroRestriction==1), diagB_0(NonZeroRestriction==1)));
    lnprioir_Omega = lnpdfIW(Omega, Psi0, nu);
    if is_Nonlinear==1
       c_0 = gamma*c_10 + (1 - gamma)*c_00;
       a_0 = gamma*a_10 + (1 - gamma)*a_00;
       lnprior_Sigma = sumc(lnpdfig(diag_Sigma, a_0/2, c_0/2));
       lnprior_q = lnpdfbeta(q, a_q0, c_q0);
    else
       lnprior_Sigma = 0;
       lnprior_q = 0;
    end
    
    lnprior = lnprior_beta + lnprior_Sigma + lnprior_q + lnprioir_Omega;
    lnPriorm(iter) = lnprior;
    
    lnpost_iter = lnL + lnprior;
    if lnpost_iter > lnPost_st
        lnPost_st = lnpost_iter;
        lnLst = lnL;
        lnPiorst = lnprior;
    end
    
end

%% 사후모드
lnPostm = lnLm + lnPriorm;

[maxlnPost, maxiter] = max(lnPostm);

lnLst = lnLm(maxiter);
lnPiorst = lnPriorm(maxiter);

disp(['lnLst = ',  num2str(lnLst)]);
disp(['lnPiorst = ',  num2str(lnPiorst)]);
disp(['lnPost_st = ',  num2str(lnLst+lnPiorst)]);
save Spec Spec
save lnLst lnLst
save lnPiorst lnPiorst
save maxiter maxiter