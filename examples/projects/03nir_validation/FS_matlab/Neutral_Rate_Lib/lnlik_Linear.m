function lnL = lnlik_Linear(Y0, YLm, beta, Omega)
% 우도함수 계산하기

[M, MMP, TP] = size(YLm);
P = MMP/(M^2);
T = TP + P;

lnL = 0;

for t = P+1:T
    y_t = Y0(t-P,:)';
    x_t = YLm(:, :, t-P);
    y_tL = x_t*beta;
    f_tL = Omega;
    f_tL = 0.5*(f_tL + f_tL');
    invf_tL = invpd(f_tL);
    invf_tL = 0.5*(invf_tL + invf_tL');
    
    lnL = lnL + lnpdfmvn1(y_t, y_tL, invf_tL);
    
end

end