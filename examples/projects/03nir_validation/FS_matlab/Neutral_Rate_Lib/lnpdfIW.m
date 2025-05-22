% https://en.wikipedia.org/wiki/Inverse-Wishart_distribution
% 
%   [pdf] = IWpdf(X, Psi, v) ;
%
% X = argument matrix, should be positive definite, p by p
% Psi = p x p symmetric, postitive definite "scale" matrix 
% v = "precision" parameter = "degrees of freeedom"
%   With this density definitions,
%   mean(X) = Psi/(v-p-1), v > p+1
%   Psi = mean(X)*(v-p-1)
%   mode(X) = Psi/(v+p+1).

function logpdf = lnpdfIW(X, Psi, v) 

p = rows(X);
logdetPsi = 0.5*v*log(det(Psi));
logExp = -0.5*trace(Psi/X);
logdetX = -0.5*(v+p+1)*log(det(X));

logConst = (v*p/2)*log(2);
logGamma = log_MGamf(p, v/2);

logpdf = logdetPsi + logExp + logdetX - logConst - logGamma;

end

% log multivariate gamma function
% https://en.wikipedia.org/wiki/Multivariate_gamma_function
function logMGF = log_MGamf(p, a)


A = (a+0.5):-0.5:(a+(1-p)/2);
GA = gamma(A);
logGA = sumc(log(GA'));
logMGF = p*(p-1)*log(pi) + logGA;

end
