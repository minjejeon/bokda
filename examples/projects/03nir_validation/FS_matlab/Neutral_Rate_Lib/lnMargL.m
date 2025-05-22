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
load maxiter

M = cols(Data);
P = Spec.P;
n1 = Spec.n1;
n0 = Spec.n0;
Ym = demeanc(Data);
[Y0, YLm] = makeYX_mex(Ym, P);
beta_0 = Spec.beta_0;
B_0 = Spec.B_0;
NonZeroRestriction = Spec.NonZeroRestriction;

%%%%%%%%%%%%%%%%%%%%%%
%% Posterior Ordinate 계산하기
%%%%%%%%%%%%%%%%%%%%%%
is_Nonlinear = Spec.is_Nonlinear;
beta_st = beta_save(maxiter, :)';
Phi_st = reshape(beta_st, M*P, M)';
diag_Sigma_st = Sigma_save(maxiter, :)';
Omega_st = reshape(Omega_save(maxiter, :)', M, M);
q_st = q_save(maxiter);
gamma_st = Gamma_save(maxiter, :)';
Mu_st = Mum(:, :, Spec.n0+maxiter);
%% beta의 밀도
disp('Computing lnpost density(beta) over MCMC cycles')
lnpdf_betam = zeros(n1, 1);
for iter = 1:n1
    
    if rem(iter, 100) == 0
        clc
        disp('Computing lnpost density(beta) over MCMC cycles')
        disp(['current iter is ', num2str(iter)])
    end
    
    Mu = Mum(:, :, n0+iter);
    Y_tilm = Ym - Mu;  % 평균제거
    [Y0_tilde, YLm_tilde] = makeYX_mex(Y_tilm, P); % VAR(P) 모형의 종속변수와 설명변수 생성
    
    Omega = reshape(Omega_save(iter, :)', M, M);
    Omega_inv = invpd(Omega);
    lnpdf_beta = lnpost_beta_mex(Y0_tilde, YLm_tilde, beta_st, P, beta_0, B_0, Omega_inv, NonZeroRestriction);
    lnpdf_betam(iter) = lnpdf_beta;

end

%%
lnpdf_beta_ = maxc(lnpdf_betam);
lnpdf_betam0 = lnpdf_betam - lnpdf_beta_;
pdf_betam0 = exp(lnpdf_betam0);
logpdf_beta0 = log(meanc(pdf_betam0));
lnpdf_beta_hat = logpdf_beta0 + lnpdf_beta_;


%% Omega의 밀도 (beta는 사후모드에 고정)
disp('Computing lnpost density(Omega) over MCMC cycles')
% 초기값
if  is_Nonlinear == 1
    Mu = Mu_st;
else
    Mu = 0*Mu_st;
end
Omega = Omega_st;
diag_Sigma = diag_Sigma_st;
gamma = gamma_st;
q =q_st;
R0 = Spec.R0;
nu = Spec.nu;
lnpdf_Omegam = zeros(n1, 1);
for iter = 1:n1
    
    if rem(iter, 100) == 0
        clc
        disp('Computing lnpost density(Omega) over MCMC cycles')
        disp(['current iter is ', num2str(iter)])
    end
    
    if  is_Nonlinear == 1
        % sampling Mu
        [Mu, G_ttm] = Gen_Mu_mex(Mu, Y0, YLm, beta_st, Phi_st, Omega, diag_Sigma, gamma);
        
        % sampling Sigma
        diag_Sigma = Gen_Sigma_mex(Mu, gamma, Spec.a_10, Spec.a_00, Spec.c_10, Spec.c_00);
        
        % sampling gamma
        gamma = Gen_Gamma_mex(diag_Sigma, q, Spec.a_10, Spec.a_00, Spec.c_10, Spec.c_00);
        
        % sampling q
        q = randbeta(1, 1, Spec.a_q0 + sumc(gamma), Spec.c_q0 + (M-sumc(gamma)));
    end
       
    Y_tilm = Ym - Mu;  % 평균제거
    [Y0_tilde, YLm_tilde] = makeYX_mex(Y_tilm, P); % VAR(P) 모형의 종속변수와 설명변수 생성
    
    % sampling Omega
    [Omega, Omega_inv] = Gen_Omega_mex(Y0_tilde, YLm_tilde, beta_st, nu, R0);
    
    % Omega_hat의 사후 밀도 계산하기
    lnpdf_OMG = lnpost_Omega_mex(Y0_tilde, YLm_tilde, beta_st, nu, R0, Omega_st);
    lnpdf_Omegam(iter) = lnpdf_OMG;
    
end
lnpdf_Omega_ = maxc(lnpdf_Omegam);
lnpdf_Omegam0 = lnpdf_Omegam - lnpdf_Omega_;
pdf_Omegam0 = exp(lnpdf_Omegam0);
lnpdf_Omega0 = log(meanc(pdf_Omegam0));
lnpdf_Omega_hat = lnpdf_Omega0 + lnpdf_Omega_;


%% Sigma의 밀도 (beta와 Omega를 사후모드에서 고정)
if  is_Nonlinear == 1  % 비선형인 경우에만 계산
    disp('Computing lnpost density(Sigma) over MCMC cycles')
    lnpdf_Sigmam = zeros(n1, 1);
    for iter = 1:n1
        
        if rem(iter, 100) == 0
            clc
            disp('Computing lnpost density(Sigma) over MCMC cycles')
            disp(['current iter is ', num2str(iter)])
        end
        
        % sampling Mu
        [Mu, G_ttm] = Gen_Mu_mex(Mu, Y0, YLm, beta_st, Phi_st, Omega_st, diag_Sigma, gamma);
        
        % sampling Sigma
        diag_Sigma = Gen_Sigma_mex(Mu, gamma, Spec.a_10, Spec.a_00, Spec.c_10, Spec.c_00);
        
        % sampling gamma
        gamma = Gen_Gamma_mex(diag_Sigma, q, Spec.a_10, Spec.a_00, Spec.c_10, Spec.c_00);
        
        % sampling q
        q = randbeta(1, 1, Spec.a_q0 + sumc(gamma), Spec.c_q0 + (M-sumc(gamma)));
        
        % Sigma_hat의 사후 밀도 계산하기
        lnpdf_Sigmam(iter) = lnpost_Sigma(Mu, gamma, Spec.a_10, Spec.a_00, Spec.c_10, Spec.c_00, diag_Sigma_st);
        
    end
    
pdf_Sigmam = exp(lnpdf_Sigmam);
lnpdf_Sigma_hat = log(meanc(pdf_Sigmam));

    %% q의 밀도 (Sigma, beta와 Omega를 사후모드에서 고정)
    disp('Computing lnpost density(q) over MCMC cycles') 
    lnpdf_qm = zeros(n1, 1);
    for iter = 1:n1
        
        if rem(iter, 100) == 0
            clc
            disp('Computing lnpost density(q) over MCMC cycles')
            disp(['current iter is ', num2str(iter)])
        end
        
        %% sampling Mu
        [Mu, G_ttm] = Gen_Mu_mex(Mu, Y0, YLm, beta_st, Phi_st, Omega_st, diag_Sigma_st, gamma);
        
        %% sampling gamma
        gamma = Gen_Gamma_mex(diag_Sigma_st, q, Spec.a_10, Spec.a_00, Spec.c_10, Spec.c_00);
        
        %% sampling q
        q = randbeta(1, 1, Spec.a_q0 + sumc(gamma), Spec.c_q0 + (M-sumc(gamma)));
        
        %% q_hat의 사후 밀도 계산하기
        lnpdf_qm(iter) = lnpdfbeta(q_st, Spec.a_q0 + sumc(gamma), Spec.c_q0 + (M-sumc(gamma)) );
    end
    
    pdf_qm = exp(lnpdf_qm);
    lnpdf_q_hat = log(meanc(pdf_qm));

else
    
    lnpdf_Sigma_hat = 0;
    lnpdf_q_hat = 0;
    
end

%%
lnpost_ordinate = lnpdf_beta_hat +  lnpdf_Omega_hat + lnpdf_Sigma_hat + lnpdf_q_hat;

%% 주변우도 계산하기

load lnLst
load lnPiorst

lnPost_st = lnLst + lnPiorst;

ln_Margnal_Lik = lnPost_st - lnpost_ordinate; % 우도값

save lnPost_st lnPost_st
save lnpdf_beta_hat lnpdf_beta_hat
save lnpdf_Omega_hat lnpdf_Omega_hat
save lnpdf_Sigma_hat lnpdf_Sigma_hat
save lnpdf_q_hat lnpdf_q_hat
save lnpost_ordinate lnpost_ordinate
save ln_Margnal_Lik ln_Margnal_Lik
save Spec Spec

 
%%
load Spec
load lnLst
load lnPiorst
load lnPost_st
load lnpdf_beta_hat
load lnpdf_Omega_hat
load lnpdf_Sigma_hat
load lnpdf_q_hat
load lnpost_ordinate
load ln_Margnal_Lik


disp(['lnLst = ', num2str(lnLst)])
disp(['lnPiorst = ', num2str(lnPiorst)])
disp(['lnPost_st = ', num2str(lnPost_st)])
disp(['lnpdf_beta_hat = ', num2str(lnpdf_beta_hat)])
disp(['lnpdf_Omega_hat = ', num2str(lnpdf_Omega_hat)])
disp(['lnpdf_Sigma_hat = ', num2str(lnpdf_Sigma_hat)])
disp(['lnpdf_q_hat = ', num2str(lnpdf_q_hat)])
disp(['lnpost_ordinate = ', num2str(lnpost_ordinate)])
disp(['ln_Margnal_Lik = ', num2str(ln_Margnal_Lik)])


if  Spec.is_Nonlinear == 1
    filename = 'Table_MargL_Nonlinear.xlsx';
else
    filename = 'Table_MargL_Linear.xlsx';
end
sheet_name = 'tabe';
writematrix('log likelihood',filename,'Sheet', sheet_name, 'Range', 'A1:A1') 
writematrix('log prior',filename,'Sheet', sheet_name, 'Range', 'A2:A2') 
writematrix('log posterior',filename,'Sheet', sheet_name, 'Range', 'A3:A3') 
writematrix('log posterior(beta)',filename,'Sheet', sheet_name, 'Range', 'A4:A4') 
writematrix('log posterior(Omega)',filename,'Sheet', sheet_name, 'Range', 'A5:A5') 
writematrix('log posterior(Sigma)',filename,'Sheet', sheet_name, 'Range', 'A6:A6') 
writematrix('log posterior(q)',filename,'Sheet', sheet_name, 'Range', 'A7:A7') 
writematrix('log posterior ordinate',filename,'Sheet', sheet_name, 'Range', 'A8:A8') 
writematrix('log margnal likelihood',filename,'Sheet', sheet_name, 'Range', 'A9:A9') 

writematrix(lnLst,filename,'Sheet', sheet_name, 'Range', 'B1:B1') 
writematrix(lnPiorst,filename,'Sheet', sheet_name, 'Range', 'B2:B2') 
writematrix(lnPost_st,filename,'Sheet', sheet_name, 'Range', 'B3:B3') 
writematrix(lnpdf_beta_hat,filename,'Sheet', sheet_name, 'Range', 'B4:B4') 
writematrix(lnpdf_Omega_hat,filename,'Sheet', sheet_name, 'Range', 'B5:B5') 
writematrix(lnpdf_Sigma_hat,filename,'Sheet', sheet_name, 'Range', 'B6:B6') 
writematrix(lnpdf_q_hat,filename,'Sheet', sheet_name, 'Range', 'B7:B7') 
writematrix(lnpost_ordinate,filename,'Sheet', sheet_name, 'Range', 'B8:B8') 
writematrix(ln_Margnal_Lik,filename,'Sheet', sheet_name, 'Range', 'B9:B9') 
    
 