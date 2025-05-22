function [ImpulseRespm, VD_Estimates, MHm, Spec] = MCMC_Recursive_BVAR(Spec)

Y = demeanc(Spec.Y);
p = Spec.p;

k = cols(Y);

% prior for inv(Omega)
nu = rows(Y)/1 + k + 2; % Following Giannone, Lenza and Primiceri (2015)
[~, Omega_OLS] = OLS_VAR(demeanc(Y(1:end, :)), 3);
Omega0 = diag(diag(Omega_OLS));
% Omega0 = Omega_OLS;
R0 = invpd(Omega0)/nu;  % Note that E(inv(Omega)) = nu*R0, Omega의 샘플링에 사용

% prior for Phi(beta)
% beta의 사전평균
AR1_ = 0.5; % AR(1)의 사전평균
Phi_1 = AR1_*eye(k);


if p > 1
    b_ = vec([Phi_1'; zeros((p-1)*k ,k)]);
else
    b_ = vec(Phi_1');
end

var_1 = 0.01*(ones(k, k) - eye(k)) + 0.25*eye(k);

if p > 1
    var_ = vec([var_1'; 0.0001*ones((p-1)*k ,k)]);
else
    var_ = vec(var_1');
end


var_  = diag(var_);


n0 = Spec.n0;
n1 = Spec.n1;
p = Spec.p;
mlag = Spec.mlag;
Variable_Names = Spec.Variable_Names;

k = cols(Y);

% 초기값
Phi = reshape(b_, p*k, k);
Omega_inv = nu*R0;

% 충격반응함수를 저장할 방
ImpulseRespm = zeros(n1,mlag+1,k^2); % (iter,j,1)은 변수 1이 변수1에 j기 이후 미치는 영향

% 축약형 모형계수를 저장할 방
pkk = p*k*k;
betam = zeros(n1,pkk);
Omegam = zeros(n1,k^2);

%% 사후분포 추출하기
[Y0, YLm] = makeYX(Y,p); % 종속변수(Y0)와 설명변수(YLm) 만들기
n = n0 + n1;

for iter = 1:n

    [~, resid] = minresid(iter,100);
    if resid == 0
        clc
        disp(['현재 반복시행은 ',num2str(iter)]);
    end

    % Phi sampling 하기
    [Phi,Fm,beta] = Gen_Phi(Y0,YLm,Phi,p,b_,var_,Omega_inv);
    
    % Omega sampling 하기
    [Omega,Omega_inv] = Gen_Omega(Y0,YLm,beta,nu,R0);

    % 충격반응함수 계산해서 저장하기
    if iter > n0
        ImpulseRespm = Gen_ImRes(Omega,Fm,mlag,n0,ImpulseRespm,iter);
        betam(iter-n0,:) = beta';
        Omegam(iter-n0,:) = vec(Omega)';
    end

end

%% 축약모형계수 추정결과
MHm = [betam, Omegam];

%% 그림 그리기
[ImpulseRespm, VD_Estimates] = Plot_IRF_VD(ImpulseRespm, Variable_Names);



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

YLm = zeros(k,kki,T-p); % 설명변수를 3차원으로 새롭게 저장할 방
for t = 1:(T-p)
    xt = kron(eye(k), YL(t,:));
    YLm(:,:,t) = xt; % p by k
end
end






%% 충격반응함수 그리기 %%%%%%%%%%%%%%%%%%%%%%%%%%
function [ImpulseRespm, VD_Estimates]= Plot_IRF_VD(ImpulseRespm, Variable_Names)

[~, mlag1, k2] = size(ImpulseRespm); % k2 = k^2, mlag1 = mlag + 1

k = sqrt(k2);
ql = [0.05;0.5;0.95]; % 5% 신뢰구간
xa = 0:(mlag1-1);

Impulse_Resp_Estimates = zeros(k, mlag1, k);

a = 1:k2;
a = reshape(a, k, k);
figure
zeroline = zeros(mlag1,1);
for i = 1:k2

    ImpulseResp_ij = ImpulseRespm(:, :, i); % n1 by (mlag+1)
    ImpulseResp_ij = quantile(ImpulseResp_ij,ql)'; % (mlag+1) by 3

    [r, c] = find(a==i);

    subplot(k, k, i);
    plot(xa, ImpulseResp_ij(:, 1), 'k--', xa, ImpulseResp_ij(:, 2), 'b-',xa, ImpulseResp_ij(:, 3), 'k--', xa, zeroline, 'k:','linewidth', 2);
    xlim([0  mlag1]);
    title([Variable_Names{c}, ' shock to ', Variable_Names{r}])

    Impulse_Resp_Estimates(c, :, r) = ImpulseResp_ij(:, 2)'; % 1 by mlag+1

end

sgtitle('Impulse Responses')


%% Variance Decomposition
VD_Estimates = zeros(k, mlag1, k);

for i = 1:k

  Impulse_Resp = Impulse_Resp_Estimates(:, :, i); % k by mlag +1
  ImpRes2 = Impulse_Resp.*Impulse_Resp; % k by mlag +1
  VD = cumsumc(ImpRes2')'; % k by mlag +1

  for j = 1:mlag1
    VD(:, j) = VD(:, j)/sumc(VD(:, j));
  end

  VD_Estimates(:, :, i) = VD;

end


figure
for vari = 1:k
    
    VD = VD_Estimates(:, :, vari);

    for shock = 1:k

        i  = a(vari, shock);
        subplot(k, k, i);
        plot(xa, VD(shock, :)', 'b-', xa, ones(mlag1, 1), 'k:', xa, zeroline, 'k:','linewidth', 2);
        xlim([0  mlag1]);
        title([Variable_Names{shock}, ' shock to ', Variable_Names{vari}])

    end
end

sgtitle('Variance Decomposition')

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
% eigF = eig(Fm); % eigenvlaue 계산
% if maxc(abs(eigF)) >= 0.99
%     Phi = Phi0;
%     Fm = [Phi'; eye((p-1)*k), zeros(k*(p-1),k)];
% end

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






%% 충격반응함수 계산하기 %%%%%%%%%%%%%%%%%%%%%%%%%%
function ImpulseRespm = Gen_ImRes(Omega,F,mlag,n0,ImpulseRespm,iter)

% B의 역행렬 계산하기
Binv = chol(Omega)';  % Lower triangular matrix

k = rows(Omega);

% 각 j에 대해서 충격반응함수 계산해서 저장하기
FF = eye(rows(F));

for j = 1:(mlag+1)

    psi_j = FF(1:k,1:k); % k by k
    theta = psi_j*Binv;  % k by k
    theta = vec(theta);  % k^2 by 1

    % 저장하기
    for i = 1:k^2
        ImpulseRespm(iter-n0,j,i) = theta(i);
    end

    FF = FF*F;

end

end





