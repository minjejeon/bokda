function lnL = lnlik(Mu, Y0, YLm, beta, Phi, Omega, diag_Sigma, gamma, is_Nonlinear)
% 우도함수 계산하기

if is_Nonlinear == 1
     lnL = lnlik_NonLinear(Mu, Y0, YLm, beta, Phi, Omega, diag_Sigma, gamma);
    
else
    lnL = lnlik_Linear(Y0, YLm, beta, Omega);
end

end