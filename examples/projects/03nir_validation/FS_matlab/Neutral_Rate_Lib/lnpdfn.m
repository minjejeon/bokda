% log pdf of normal
% x = normal variates
% mu = vector of means
% sig2vec = vector of variances
function [retf] = lnpdfn(x,mu,sig2vec)

 c = -0.5*log(2*sig2vec*pi); 
  e = x - mu;
  e2 = e.*e;

  retf = c - 0.5*e2./sig2vec;
end
