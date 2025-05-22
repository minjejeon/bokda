% X and Mu are both D by 1 vectors, and Sigma is a D by D matrix. 
%
%   Y = lnpdfmvn4(X,MU,Chol_SIGMA) 
% mu = mean
% Chol_SIGMA = Chol(Sigma): upper triangular
% This is a modified version of MVNPDF.
function y = lnpdfmvn4(X, Mu, Chol_Sigma)

X0 = X - Mu;
d = rows(X);
R = Chol_Sigma';
logSqrtDetSigma = sum(log(diag(R)));
xRinv = R\X0;
quadform = sum(xRinv.^2, 1);

y = -0.5*quadform - logSqrtDetSigma - d*log(2*pi)/2;
