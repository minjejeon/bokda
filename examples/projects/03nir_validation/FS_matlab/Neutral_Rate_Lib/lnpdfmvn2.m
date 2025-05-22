% /* log pdf of multivariate normal */
% /* C = chol(precision) */
function [retf] = lnpdfmvn2(y,mu,C)

%            clc;
%            disp(size(y));
%            disp(size(mu));
%            disp(size(C));

e = C*(y-mu);          %  /* standard normals: k times m matrix */

retf = 0.5*lndet1(C) + sumc(lnpdfn1(e)); %  /* the log of the density */

end