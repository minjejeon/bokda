function [Mu, G_ttm] = Gen_Mu(Mu, Y0, YLm, beta, Phi, Omega, diag_Sigma, gamma)


[M, MMP, TP] = size(YLm);
P = MMP/(M^2);
T = TP + P;
MP1 = M*(P+1);

for m = 1:M
    
    if gamma(m) == 0
        Mu(:, m) = zeros(T, 1);
    end
end
% Mu(1:P, :) = zeros(P, M);

% 상태변수의 초기값은 주어진 것으로 처리
G_LL = zeros(MP1, 1);
for p = P+1:(-1):1
    G_LL(M*(p-1)+1:M*p) = Mu(p, :)';
end

P_LL = 0.1*eye(MP1);

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
%     G_ttm(1:P, :) = kron(ones(P, P+1), Mu(1, :));
%     for p = 1:(P+1)
%         G_ttm(p:end, (p-1)*M+1:p*M) = Mu(1:end+1-p, :);
%     end
    
    P_ttm = zeros(MP1, MP1, T);

    G_tLm = zeros(T, MP1);   % predictive values
    P_tLm = zeros(MP1, MP1, T);

    
    for t = P+1:T
        y_t = Y0(t-P,:)';
        x_t = YLm(:, :, t-P);
        G_tL = F*G_LL;
        P_tL = F*P_LL*F' + SIGMA;
        y_tL = x_t*beta + X * G_tL;
        f_tL = X*P_tL * X' + Omega;
        invf_tL = inv(f_tL);
        G_tt = G_tL + P_tL * X' * invf_tL * (y_t - y_tL);
        P_tt = P_tL - P_tL * X' * invf_tL * X * P_tL;

        G_tLm(t, :) = G_tL'; % save for use in backward recursion
        P_tLm(:,:, t) = P_tL;

        G_ttm(t, :) = G_tt';
        P_ttm(:, :, t) = P_tt;

        G_LL = G_tt;
        P_LL = P_tt;
    end


    %
    % backward recursion

    G_TT = G_ttm(T, :)'; % M*(P+1) by 1
    P_TT = P_ttm(:, :, T); % M*(P+1) by M*(P+1)

    G_t1 = G_TT + chol(P_TT)'*randn(MP1, 1); % G(t+1)
    Mu(T, :) = G_t1(1:M)';

    for t = T-1:(-1):P+1

        P_tt = P_ttm(:, :, t);
        P_tL = P_tLm(:, :, t+1);
        G_tL = G_tLm(t+1, :)';
        P_tLinv = inv(P_tL);
        G_tt1 = G_ttm(t,:)' + P_tt*F'*P_tLinv*(G_t1 - G_tL);
        P_tt1 = P_tt - P_tt*F'*P_tLinv*F *P_tt;

        P_tt1 = (P_tt1 + P_tt1')/2;  
        G_t = G_tt1 + cholmod(P_tt1)'*randn(MP1, 1);

        Mu(t,:) = G_t(1:M)';

        G_t1 = G_t;
    end
end