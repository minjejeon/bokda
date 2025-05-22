%% sigma 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%
function sigma = Gen_Sigma(Um, B, Cbar, a0, sigma_)

M = cols(B);
T = rows(Um);

Yhatm = Um(:, 1:M);
C1 = Cbar(:, 1:M);
C2 = Cbar(:, M+1:end);

C1inv = inv(C1);
Ehatm = zeros(T, M);
for t = 2:T
    xt = Um(t, M+1:end)' - Um(t-1, M+1:end)';
    Ehat = Yhatm(t, :)' - B*Yhatm(t-1, :)' - C2*xt; % 3 by 1
    Ehatm(t, :) = C1inv*Ehat;
end

sigma = eye(M);
for m = 1:M
   sigma(m, m) = Gen_sig2(Ehatm(:, m), zeros(T, 1), 0, a0, a0*sigma_(m));
end

end