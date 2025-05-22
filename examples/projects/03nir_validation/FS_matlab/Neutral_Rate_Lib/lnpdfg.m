% /* log pdf of gamma density */
% /* a : shape parameter */ 
% /* b : inverse scale parameter */
% /* mean = a/b, var = a/(b^2) */
function [retf] = lnpdfg(x,a,b)

gam = gamma(a);
retf = a.*log(b) + (a-1).*log(x) - b.*x - log(gam);

end