function lnpost_pdf = lnpost_Sigma(Mu, gamma, a_10, a_00, c_10, c_00, diag_Sigma_st)

scale = 1000;
Mu = Mu(1:end, :); % 번인 제거
[T, M] = size(Mu);
lnpost_pdfm = zeros(M, 1);
      
    for m = 1:M
        
        ehat = Mu(2:end, m) - Mu(1:end-1, m); % 잔차항, T-1 by 1
        ehat2 = ehat'*ehat; % 1 by 1
        c_0 = c_10*gamma(m) + c_00*(1 - gamma(m));
        a_0 = a_10*gamma(m) + a_00*(1 - gamma(m));
         
        lnpost_pdfm(m) = lnpdfig(scale*diag_Sigma_st(m), (a_0 + T)/2, scale*(c_0 + ehat2)/2);
        
        
    end
    
    lnpost_pdf = sumc(lnpost_pdfm);
end