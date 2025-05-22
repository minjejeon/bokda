function [phi_hat, sig2_hat] = OLS_ARp_woC(Y, p)

y0 = Y(p+1:end); % T-p by k
T = rows(y0);

y_lag = [ ];

for j = 1:p
  y_lag = [y_lag, Y(p+1-j:end-j)]; 
end

y_lag2 = y_lag'*y_lag;
phi_hat = y_lag2\(y_lag'*y0); % p*k by k
 
% Omega_hat 
u_hat = y0 - y_lag*phi_hat; % T-p by k 
sig2_hat = u_hat'*u_hat/(T-p);  % k by k

end