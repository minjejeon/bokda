%% Functions
function [beta_0, B_0, Omega_hat, V_mat] = Minnesota_prior(y, P, AR0, lambda_1,lambda_2, mKOR)

y = demeanc(y); % 평균제거
[T,m] = size(y);

% Step 0: Calculate prior mean
Phi_1 = AR0*eye(m);
beta_0 = vec([Phi_1,zeros(m,m*(P-1))]');

% Step 1: Run VAR to derive standard error for each equation
Y0 = y(1+P:end,:);
X = zeros(T-P,m*P);
for i = 1:P
    X(:,m*(i-1)+1:m*i) = y(1+P-i:end-i,:);
end
beta_ols = inv(X'*X)*X'*Y0;
resid = Y0 - X*beta_ols;
Omega_hat = resid'*resid/(T-P);
sig = sqrt(diag(Omega_hat));

% Step 2: Calculate standard error
V_mat = zeros(m,m*P);
for p = 1:P
    Vp = zeros(m,m);
    for i = 1:m
        for j = 1:m
            if i==j
                Vp(i,j) = lambda_1/p^2;
            else
                Vp(i,j) = lambda_1*lambda_2*(sig(i)/sig(j))/p^2;
            end

            if mKOR < m
            for k = (mKOR+1):m
                   for s = 1:mKOR
                       Vp(k, s) = 0.00000001*(sig(k)/sig(s));
                   end
            end
            end
            
        end
    end
V_mat(:,(p-1)*m+1:p*m) = Vp;
end

vec_ = reshape(V_mat',m^2*P,1);
B_0 = diag(vec_);

end






