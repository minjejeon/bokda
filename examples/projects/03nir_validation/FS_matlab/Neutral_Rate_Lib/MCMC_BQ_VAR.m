function [ImpulseRespm, MHm]= MCMC_BQ_VAR(n0, n1, Spec)

b_ = Spec.b_;
var_ = Spec.var_;
p = Spec.p;
nu = Spec.nu;
R0 = Spec.R0;
Y = Spec.Y;
J = Spec.J;

k = cols(Y);

% 초기값
Phi = reshape(b_, p*k, k);
Omega_inv = nu*R0;

% 충격반응함수를 저장할 방
ImpulseRespm = zeros(n1,J+1,k^2); % (iter,j,1)은 변수 1이 변수1에 j기 이후 미치는 영향 

% 축약형 모형계수를 저장할 방
pkk = p*k*k;
betam = zeros(n1,pkk);
Omegam = zeros(n1,k^2);

%% 사후분포 추출하기
[Y0, YLm] = makeYX(Y,p); % 종속변수(Y0)와 설명변수(YLm) 만들기
n = n0 + n1;

for iter = 1:n

    counter_iter(iter)
        
    % Phi sampling 하기
    [Phi,Fm,beta] = Gen_Phi(Y0,YLm,Phi,p,b_,var_,Omega_inv);
    
    % Omega sampling 하기
    [Omega,Omega_inv] = Gen_Omega(Y0,YLm,beta,nu,R0);
    
    % 충격반응함수 계산해서 저장하기
    if iter > n0
        ImpulseRespm = Gen_ImRes(Omega,Fm,J,n0,ImpulseRespm,iter); 
        betam(iter-n0,:) = beta';
        Omegam(iter-n0,:) = vec(Omega)';
    end
    
end

%% 축약모형계수 추정결과
MHm = [betam Omegam];

%% 그림 그리기
Plot_IRF(ImpulseRespm);
end

%% 종속, 설명변수 만들기 %%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y0,YLm] = makeYX(Y,p)

k = cols(Y); % 변수의 수
T = rows(Y); % 시계열의 크기

Y0 = Y(p+1:T,:); % 종속변수

    % 설명변수(=Y의 과거값) 만들기
    YL = zeros(T-p,p*k);
    for i = 1:p
        YL(:,k*(i-1)+1:k*i) = Y(p+1 - i:T-i,:); 
    end
    
    ki = p*k; % 각 식에 있는 설명변수의 수
    kki = k*ki;

    YLm = zeros(k, kki, T-p); % 설명변수를 3차원으로 새롭게 저장할 방
    for t = 1:(T-p)
        xt = kron(eye(k), YL(t,:));
        YLm(:,:,t) = xt; % p by k
    end
end

%% 충격반응함수 그리기 %%%%%%%%%%%%%%%%%%%%%%%%%%
function Plot_IRF(ImpulseRespm)

[~, mlag1, k2] = size(ImpulseRespm); % k2 = k^2, mlag1 = J + 1

k = sqrt(k2);
ql = [0.05;0.5;0.95]; % 5% 신뢰구간
xa = 0:(mlag1-1);

a = 1:k2;
a = reshape(a, k, k);
% 충격반응함수
figure
zeroline = zeros(mlag1,1);
for i = 1:k2
    
    ImpulseResp_ij = ImpulseRespm(:,:,i); % n1 by (J+1)
    ImpulseResp_ij = quantile(ImpulseResp_ij,ql)'; % (J+1) by 3
    
    [r,c] = find(a==i);
    
    subplot(k,k,i); 
    plot(xa, ImpulseResp_ij(:, 1), 'k--', xa, ImpulseResp_ij(:, 2), 'b-',xa, ImpulseResp_ij(:, 3), 'k--', xa, zeroline, 'k:','linewidth', 2);
    xlim([0  mlag1]);
    title(['shock ',num2str(c) , ' to vari ', num2str(r)])
    sgtitle('Impulse Responses')
end

% 누적충격반응함수
figure
zeroline = zeros(mlag1,1);
for i = 1:k2
    
    ImpulseResp_ij = ImpulseRespm(:, :, i);% n1 by (J+1)
    
    Cum_ImpulseResp_ij = cumsum(ImpulseResp_ij')'; % 충격반응함수 누적하기
    
    Cum_ImpulseResp_ij = quantile(Cum_ImpulseResp_ij, ql)'; % (J+1) by 3
    
    [r,c] = find(a==i);
    
    subplot(k,k,i); 
    plot(xa, Cum_ImpulseResp_ij(:, 1), 'k--', xa, Cum_ImpulseResp_ij(:, 2), 'b-',xa, Cum_ImpulseResp_ij(:, 3), 'k--', xa, zeroline, 'k:','linewidth', 2);
    xlim([0  mlag1]);
    title(['shock ',num2str(c) , ' to vari ', num2str(r)])
    sgtitle('Cumulative Impulse Responses')
end


end


%% Phi 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%
function [Phi,Fm,beta] = Gen_Phi(Y0,YLm,Phi0,p,b_,var_,Omega_inv)

    X = YLm; % 설명변수, 3차원
    XX = 0;
    XY = 0;
    T0 = rows(Y0); % = T-p
    k = cols(Y0);
    
    for t = 1:T0
        Xt = X(:,:,t);
        XX = XX + Xt'*Omega_inv*Xt;
        XY = XY + Xt'*Omega_inv*Y0(t,:)';        
    end

    precb_ = invpd(var_); 
    B1_inv = precb_ + XX;
    B1_inv = 0.5*(B1_inv + B1_inv');
    B1 = invpd(B1_inv);
    B1 = 0.5*(B1 + B1');
    A = XY + precb_*b_; % b_ = B0
    BA = B1*A; % full conditional mean  

    Chol_B1 = cholmod(B1)';
    beta = BA + Chol_B1*randn(p*k*k,1); % beta sampling 하기

    % F 행렬만들기
    Phi = reshape(beta,p*k,k);  % p*k by k
    Fm = [Phi'; eye((p-1)*k), zeros(k*(p-1),k)]; % p*k by p*k
   
    % 안정성 확인하기
    eigF = eig(Fm); % eigenvlaue 계산
    if maxc(abs(eigF)) >= 1 
        Phi = Phi0;
        Fm = [Phi'; eye((p-1)*k), zeros(k*(p-1),k)];
    end
    
end    








%% Omega 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%
function [Omega,Omega_inv] = Gen_Omega(Y,X,beta,nu,R0)
   
T = rows(Y);
k = cols(Y);

ehat2 = zeros(k,k); % 잔차항의 제곱의 합

for t = 1:T
     Xt = X(:,:,t);
     ehat = Y(t,:)' - Xt*beta; % 잔차항
     ehat2 = ehat2 + ehat*ehat'; % k by k
end
        
 Omega1_inv = ehat2 + invpd(R0);
 Omega1 = invpd(Omega1_inv);
 Omega_inv = randwishart(Omega1,(T+nu));
 Omega = invpd(Omega_inv);
  
end


%% 충격반응함수 계산하기 
function ImpulseRespm = Gen_ImRes(Omega,F,J,n0,ImpulseRespm,iter)

% B의 역행렬 계산하기
pk = length(F);
k = length(Omega);
Psi_ = inv(eye(pk)-F);
Psi_ = Psi_(1:k,1:k);

theta_ = chol(Psi_*Omega*Psi_')';

% The Long-run effect of demand shock to unemployment rate should be negative
theta_(2,2) = - theta_(2,2); 

inv_B = Psi_\theta_;

k = rows(Omega);

% 각 j에 대해서 충격반응함수 계산해서 저장하기
FF = eye(rows(F));

 for j = 1:(J+1)
     
       psi_j = FF(1:k,1:k); % k by k 
       theta = psi_j*inv_B;  % k by k 
       theta = vec(theta);  % k^2 by 1
       
       % 저장하기
       for i = 1:k^2
           ImpulseRespm(iter-n0, j, i) = theta(i);
       end
       
       FF = FF*F;
       
 end
 
end





