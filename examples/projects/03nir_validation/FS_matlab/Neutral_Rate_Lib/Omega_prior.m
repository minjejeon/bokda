function R_0 = Omega_prior(y,p,nu_0)
% Returns scale parameter for Omega, which implies prior mean of variance estimated
% from OLS

[T,m] = size(y);

X = zeros(T-p,p);
sig2_hat = zeros(m,1);
for k = 1:m
    Y0 = y(p+1:end,k);
    for i = 1:p
        X(:,i) = y(p+1-i:end-i,k);
    end
    beta_ols = inv(X'*X)*X'*Y0;
    Y0_hat = X*beta_ols;
    resid = Y0 - Y0_hat;
    sig2_hat(k) = resid'*resid/(T-p);
end
Omega_0 = diag(sig2_hat);
R_0 = Omega_0 * (nu_0 - m -1);
end

