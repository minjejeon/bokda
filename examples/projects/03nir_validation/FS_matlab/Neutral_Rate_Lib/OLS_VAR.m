function [phi_hat, Omega_hat, F] = OLS_VAR(Y, p)

y = demeanc(Y);
[T, k] = size(y);
y0 = y(p+1:T,:); % T-p by k

y_lag = [ ];

for j = 1:p
  y_lag = [y_lag y(p+1-j:T-j,:)]; 
end

y_lag2 = y_lag'*y_lag;
phi_hat = y_lag2\(y_lag'*y0); % p*k by k
 
if p > 1
  F = [phi_hat'; eye((p-1)*k), zeros(k*(p-1),k)];  % p*k by p*k
elseif p==1
  F = phi_hat';
end

T = rows(y0);

% Omega_hat 
u_hat = (y0 - y_lag*phi_hat); % T-p by k 
Omega_hat = u_hat'*u_hat/(T-p*k);  % k by k

end