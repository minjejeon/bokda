function beta = Gen_rho(Ym, Trend, Cycle, b_0, V_rho, Omega)

precB_0 = inv(V_rho);
sig2 = Omega(4,4);

Y = Ym(2:end, 4) - Cycle(2:end, 4) - Trend(1:end-1, 4);
X = Trend(2:end, 1:2);

k = cols(X);
XX = X'*X;
XY = X'*Y;

B_1 = invpd((1/sig2)*XX + precB_0); % full conditional variance B_1
A = (1/sig2)*XY + precB_0*b_0;
M = B_1*A;

beta = M + chol(B_1)'*randn(k, 1); % sampling beta


end