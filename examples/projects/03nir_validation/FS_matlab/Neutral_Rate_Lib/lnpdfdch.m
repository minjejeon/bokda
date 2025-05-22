% log Dirichlet density
% x = (k-1) by 1
% am =  k by 1
function [lnpdf] = lnpdfdch(x,am)

x1 = 1 - sumc(x);
x = [x;x1];
lnpdf = gammaln(sumc(am)) - sumc(gammaln(am));
lnpdf = lnpdf + sumc((am-1).*log(x));

end

