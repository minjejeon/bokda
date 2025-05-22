%% B 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%
function [Phi, Sigma, Sigma_inv] = Gen_Phi_Sigma(Phi0, Cycle, b_, var_, Sigma_inv,nu,R0)

[Y0,YLm] = makeYX(Cycle,1);

%%
[Phi, ~, beta] = Gen_Phi(Y0,YLm,Phi0, 1, b_, var_, Sigma_inv);

if minc(isfinite(beta)) == 1 
    %안정성 확인하기
    eigF = eig(Phi); % eigenvlaue 계산
    if maxc(abs(eigF)) >= 0.98
        %     disp(maxc(abs(eigF)))
        Phi = Phi0;
    end
    
else
    Phi = Phi0;
    
end
%%
 [Sigma, Sigma_inv] = Gen_Sigma(Y0,YLm,beta,nu,R0);

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
function [Omega,Omega_inv] = Gen_Sigma(Y,X,beta,nu,R0)
   
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