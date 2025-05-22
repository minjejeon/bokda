function gamma = Gen_Gamma(diag_Sigma, q, a_10, a_00, c_10, c_00)

scale = 1000;
diag_Sigma = scale*diag_Sigma;
c_10 = scale*c_10;
c_00 = scale*c_00;
M = rows(diag_Sigma);
gamma = zeros(M, 1);

    for m = 1:M
        sig2 = diag_Sigma(m);
        density_1 = exp(lnpdfig(sig2, a_10/2, c_10/2));
        density_0 = exp(lnpdfig(sig2, a_00/2, c_00/2));
        prb1 = (density_1*q) / (density_1*q + density_0*(1-q));
        gamma(m) = rand(1,1) < prb1;
    end
    
end