% log pdf of normal
% x = normal variates
% mu = vector of means
% prec = vector of precision
function [retf] = lnpdfn2(x,mu,prec)

 c = 0.5*log(2*prec*pi);  
  e = x - mu;
  e2 = e.*e;

  retf = c - 0.5*e2.*prec;
end
