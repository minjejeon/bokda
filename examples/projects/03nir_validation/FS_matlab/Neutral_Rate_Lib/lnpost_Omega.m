%% Omega 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%
function lnpost_pdf = lnpost_Omega(Y, X, beta, nu, R0, Omega_st)

T = rows(Y);
k = cols(Y);

ehat2 = zeros(k, k); % 잔차항의 제곱의 합

for t = 1:T
    Xt = X(:,:,t);
    ehat = Y(t, :)' - Xt*beta; % 잔차항
    ehat2 = ehat2 + ehat*ehat'; % k by k
end

R1inv = ehat2 + invpd(R0);
R1 = invpd(R1inv);
nu1 = nu + T;

Omega1_inv = R1*nu1;
Omega1_inv = 0.5*(Omega1_inv + Omega1_inv');
Psi1 = invpd(Omega1_inv)*nu1;

% R0 = Omega0_inv/nu; 
% Omega0_inv = R0*nu;
% Psi0 = Omega0*nu;

lnpost_pdf = lnpdfIW(Omega_st, Psi1, nu1);
end