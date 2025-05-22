function [Cbar, accept] = Gen_C(Um, B, Cbar0, omega, sigma, C_, V_)

M = rows(omega);
K = 2*M;

indMH = [2;3;4;6;7;8;10;11;12;13;14;15];

vecCbar0 = vec(Cbar0);
vecCbar1 = vecCbar0;
vecCbar1(indMH, :) = vecCbar0(indMH, :) + 0.02*randn(rows(indMH), 1);
Cbar1 = reshape(vecCbar1, M, K);

valid = paramconst(Cbar1);

if valid == 1
lnprior0 = sumc(lnpdfn(vecCbar0, vec(C_), vec(V_)));
lnprior1 = sumc(lnpdfn(vecCbar1, vec(C_), vec(V_)));

lnalpha = lnlik(Um, B, Cbar1, sigma, omega) + lnprior1 ...
            - lnlik(Um, B, Cbar0, sigma, omega) - lnprior0;

if log(rand(1,1)) < lnalpha
    Cbar = Cbar1;
    accept = 1;
else
    Cbar = Cbar0;
    accept = 0;
end

else
    Cbar = Cbar0;
    accept = 0;
end
end

%%
function lnL = lnlik(Um, B, Cbar, sigma, omega)

M = cols(Um)/2;
T = rows(Um);
zero1 = zeros(M, 1);
zero1(end) = 1;

Omega = eye(cols(Cbar));
Omega(1:M, 1:M) = sigma;
Omega(M+1:end, M+1:end) = omega;
OMEGA = Cbar*Omega*Cbar';
Yhatm = Um(:, 1:M);
xrm = Um(2:end, 6) - Um(1:end-1, 6);
xrm = [0; xrm];
lnL = 0;
for t = 2:T
    Ehat = Yhatm(t, :)' - B*Yhatm(t-1, :)' + zero1*xrm(t); % 3 by 1
    lnp = lnpdfmvn(Ehat, zeros(M, 1), OMEGA);
    lnL = lnL + lnp;
end

end

%%
function valid = paramconst(Cbar)

validm = ones(10, 1);

validm(1) = Cbar(2,1) > 0;
validm(2) = Cbar(1,2) < 0;
validm(3) = Cbar(1,3) < 0;
validm(4) = Cbar(2,3) < 0;
isfinitem = isfinite(Cbar);
validm(5) = minc1(isfinitem) == 1;


valid = minc(validm);

end