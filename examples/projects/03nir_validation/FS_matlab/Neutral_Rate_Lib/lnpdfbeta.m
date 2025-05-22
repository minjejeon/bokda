%  log pdf of beta dist
function [retf] =  lnpdfbeta(p,a,b)

retf = gammaln(a+b) - gammaln(a) - gammaln(b);
retf = retf + (a-1).*log(p) + (b-1).*log(1-p);

end