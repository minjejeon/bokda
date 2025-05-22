function sig2 = Gen_sig2( Y, X, beta, a_0, d_0 )

T = rows(X);
a_1 = T + a_0;
e = Y - X*beta;
d_1 = d_0 + e'*e;

sig2 = randig(a_1/2, d_1/2, 1, 1); % sig2 sampling


end
