Spec.n0 = n0;
Spec.n1 = n1;
Spec.P = P;
Spec.H = H;

%%
Vari_index = Vari_index0(is_included==1);
is_KOR = is_KOR0(is_included==1);

is_growth = is_growth0(is_included==1);
Vari_Names_ENG = Vari_Names_ENG0(is_included==1); % 변수이름 (한글)
Vari_Names_KOR = Vari_Names_KOR0(is_included==1); % 변수이름 (영문)
Raw_Data = Raw_Data0(:, is_included==1);

Vari_Names = Vari_Names_KOR;


M = cols(Raw_Data);   % 총변수의 수
mKOR = sumc(is_KOR); % 국내변수의 수
Spec.mKOR = mKOR;  
Spec.is_Nonlinear = is_Nonlinear;
Spec.is_MargL = is_MargL;
Spec.AR0 = AR0;
Spec.lambda_1 = lambda_1;
Spec.lambda_2 = lambda_2;
Data = [ ];
%% 자료변환
for vari = 1:M
    
    if is_growth(vari) == 1
        data = 100*(Raw_Data(5:end, vari)  -  Raw_Data(1:end-4, vari)) ./ Raw_Data(1:end-4, vari);
    else
        data = Raw_Data(5:end, vari);
    end
    
    Data = [Data, data];
    
end

Ym = demeanc(Data);
Ymean = meanc(Data);
[T, M] = size(Ym);

xtick = 5:20:T;
xticklabel = {'04:Q1', '07:Q1', '10:Q1', '13:Q1', '16:Q1', '19:Q1'};

%% 추정옵션: 0일 경우 해당 파라미터는 샘플링되지 않는다
is_beta_sampling = 1;
is_omega_sampling = 1;

is_sigma_sampling = 1;
is_gamma_sampling = 1;
is_q_sampling = 1;

is_Mu_sampling = 1;
is_forecasting = 1;

%% 사전분포 및 초기값


%% 상태변수 초기값 (mu_t, t=1,...,T)
[~, time_Trend, bm, tm]  = detrend(Ym);

if is_Nonlinear == 1
    
    Mu = time_Trend;
    % 평균 시변 여부 (gamma)
    gamma = ones(M, 1);
    for m = 1:M
        if abs(tm(2, m)) < 1
            gamma(m) = 0;
            Mu(:, m) = 0;
        end
    end
    
elseif is_Nonlinear == 0
    
    Mu = zeros(T, M);
    gamma = zeros(M, 1);
    diag_Sigma = zeros(M, 1);
    
end

save time_Trend time_Trend


%% Minnesota_prior
Y_tilm = Ym - time_Trend;  % 평균제거
[beta_0, B_0, Omega_OLS_hat, V_mat] = Minnesota_prior(Y_tilm, P, AR0, lambda_1, lambda_2, mKOR);

Spec.beta_0 = beta_0;
Spec.B_0 = B_0;



%% VAR로 변수별 충격 추정하고 저장하기 (시나리오분석)
SE = sqrt(diag(Omega_OLS_hat));
filename = 'Scenario.xlsx';
sheetname = 'Scenario';

writematrix(Vari_index',filename,'Sheet', sheetname, 'Range', 'D4:D50') 
writecell(Vari_Names',filename,'Sheet', sheetname, 'Range', 'E4:E50') 
writematrix('변수',filename,'Sheet', sheetname, 'Range', 'E1:E1') 
writematrix('충격지속기간',filename,'Sheet', sheetname, 'Range', 'E3:E3') 
writematrix('표준편차',filename,'Sheet', sheetname, 'Range', 'F1:F1')
writematrix(SE,filename,'Sheet', sheetname, 'Range', 'F4:F50')
writematrix('주의단계',filename,'Sheet', sheetname, 'Range', 'G1:G1')
writematrix('심각단계',filename,'Sheet', sheetname, 'Range', 'H1:H1')
writematrix('위기단계',filename,'Sheet', sheetname, 'Range', 'I1:I1')



%%
beta = beta_0; % 초기값
Phi0 = reshape(beta_0, M*P, M)';
Phi = Phi0;

% 오차항 공분산 (Omega)
% 각 변수에 대한 VAR(P) 모형의 OLS 오차항 분산 추정치를 사용한다
Omega0 = diag(diag(Omega_OLS_hat));
Omega0 = 0.5*(Omega0 + Omega0');
Omega0_inv = invpd(Omega0);

nu = M+2;  % 자유도 > (M+1)
R0 = Omega0_inv/nu;  % Note that E(inv(Omega)) = nu*R0, Omega의 샘플링에 사용

Spec.nu = nu;
Spec.R0 = R0;
Spec.Omega0 = Omega0;

Omega_inv = Omega0_inv; % 초기값

%
c_00 = b0*a_00;
c_10 = b1*a_10;

Spec.c_00 = c_00;
Spec.c_10 = c_10;
Spec.a_00 = a_00;
Spec.a_10 = a_10;

% 추세 존재 확률 (q)
a_q0 = 5; c_q0 = 5;

Spec.a_q0 = a_q0;
Spec.c_q0 = c_q0;

q = a_q0/(a_q0 + c_q0);

%% 저장할 방
beta_save = zeros(n, (M^2)*P);
Omega_save = zeros(n, M^2);
Sigma_save = zeros(n, M);
q_save = zeros(n, 1);
Gamma_save = zeros(n, M);

ns_counter = 0; % 샘플링된 베타가 stationarity test 통과 못하는 경우

%% 깁스-샘플링
Mum = zeros(T, M, n);
YFm = zeros(n, H, M);
[Y0, YLm] = makeYX_mex(Ym, P);

ind_Vari = 1:M;
ind_Vari = ind_Vari';
for iter=1:n
    
    %% 베타 샘플링
    Y_tilm = Ym - Mu;  % 평균제거
    [Y0_tilde, YLm_tilde] = makeYX_mex(Y_tilm, P); % VAR(P) 모형의 종속변수와 설명변수 생성
    
    if is_beta_sampling == 1
        [beta_0, B_0] = Minnesota_prior(Y_tilm, P, AR0, lambda_1, lambda_2, mKOR); % 미네소타 사전분포는 매기 변한다
        [Phi, Fm, beta, is_reject] = Gen_Phi_mex(Y0_tilde, YLm_tilde, Phi0, P, beta_0, B_0, Omega_inv);
        Phi0 = Phi;
        ns_counter = ns_counter + is_reject;
        beta_save(iter, :) =  beta';
    end
    
    %% 오메가 샘플링
    if is_omega_sampling == 1
        [Omega, Omega_inv] = Gen_Omega_mex(Y0_tilde, YLm_tilde, beta, nu, R0);
        Omega_save(iter, :) = vec(Omega)';
    end
    
    %% Sigma 샘플링
    if is_sigma_sampling == 1 && is_Nonlinear == 1
        diag_Sigma = Gen_Sigma_mex(Mu, gamma, a_10, a_00, c_10, c_00);
        Sigma_save(iter, :) = diag_Sigma';
    end
    
    %% q 샘플링
    if  is_q_sampling == 1 && is_Nonlinear == 1
        q = randbeta(1, 1, a_q0 + sumc(gamma), c_q0 + (M-sumc(gamma)));
        q_save(iter) = q;
    end
    
    %% 감마 샘플링
    if  is_gamma_sampling == 1 && is_Nonlinear == 1
        gamma = Gen_Gamma_mex(diag_Sigma, q, a_10, a_00, c_10, c_00);
        Gamma_save(iter, :) = gamma';
    end
    
    %% Mu 샘플링
    if  is_Mu_sampling == 1 && is_Nonlinear == 1
        [Mu, G_ttm] = Gen_Mu_mex(Mu, Y0, YLm, beta, Phi, Omega, diag_Sigma, gamma);
        Mum(:, :, iter) = Mu;
    end
    
    %% 예측하기
    if  is_forecasting == 1
        YFm = Forecasting(is_Nonlinear, YFm, Mu, Ym, beta, Omega, diag_Sigma, H, iter);
    end
    
    %% 중간결과보기
    if rem(iter,freq) == 0
        clc
        disp(['현재 반복시행은 ', num2str(iter)]);
        disp('   Vari      Phi(1)    sigma    gamma   Omega')
        disp([ind_Vari, diag(Phi(:, 1:M)), diag_Sigma, gamma, diag(Omega)])
        disp(['is_reject = ', num2str(is_reject)]);
        disp([' # of rejections = ', num2str(ns_counter)]);
    end
     
end

%% 번인 버리기
Gamma_save = Gamma_save(n0+1:end, :);
Sigma_save = Sigma_save(n0+1:end, :);
Omega_save = Omega_save(n0+1:end, :);
beta_save = beta_save(n0+1:end, :);
q_save =q_save(n0+1:end, :);

save Vari_index Vari_index
save date date
save Data Data
save Mum Mum
save Gamma_save Gamma_save
save Sigma_save Sigma_save
save Omega_save Omega_save
save beta_save beta_save
save q_save q_save
save Vari_Names Vari_Names

Spec.xtick = xtick;
Spec.xticklabel = xticklabel;

save YFm YFm
save ns_counter ns_counter
save Spec Spec