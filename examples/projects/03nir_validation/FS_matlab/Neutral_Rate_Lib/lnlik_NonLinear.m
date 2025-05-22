function lnL = lnlik_NonLinear(Mu, Y0, YLm, beta, Phi, Omega, diag_Sigma, gamma)
% 우도함수 계산하기

[M, MMP, TP] = size(YLm);
P = MMP/(M^2);
T = TP + P;
MP1 = M*(P+1);

for m = 1:M
    
    if gamma(m) == 0
        Mu(:, m) = zeros(T, 1);
    end
end

% 상태변수의 초기값은 주어진 것으로 처리
G_LL = zeros(MP1, 1);
for p = P+1:(-1):1
    G_LL(M*(p-1)+1:M*p) = Mu(p, :)';
end

P_LL = 0.00001*eye(MP1);

F = [eye(M), zeros(M,M*P); eye(M*P), zeros(M*P,M)];
W = [eye(M); zeros(M*P,M)];

Sigma = diag(diag_Sigma); % M by M
SIGMA = W*Sigma*W';

X_mat = eye(MP1);

X = X_mat(1:M, :);
for p = 1:P
    X = X - Phi(:, 1+M*(p-1):M*p)*X_mat(1+M*p:M*(p+1), :);
end

G_ttm = zeros(T, MP1);  % filtered values
P_ttm = zeros(MP1, MP1, T);

lnL = 0;

for t = P+1:T
    
    y_t = Y0(t-P,:)';
    x_t = YLm(:, :, t-P);
    G_tL = F*G_LL;
    P_tL = F*P_LL*F' + SIGMA;
    y_tL = x_t*beta + X*G_tL;
    f_tL = X*P_tL*X' + Omega;
    f_tL = 0.5*(f_tL + f_tL');
    invf_tL = invpd(f_tL);
    invf_tL = 0.5*(invf_tL + invf_tL');
    
    lnL = lnL + lnpdfmvn1(y_t, y_tL, invf_tL);
    
    Kalman_gain = P_tL*X'*invf_tL;
    G_tt = G_tL + Kalman_gain*(y_t - y_tL);
    P_tt = P_tL - Kalman_gain*X*P_tL;
    
    G_ttm(t, :) = G_tt';
    P_ttm(:, :, t) = P_tt;
    
    G_LL = G_tt;
    P_LL = P_tt;
    
end

end