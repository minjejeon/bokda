% log of multivariate t density 
% cP = chol(P), P = precision matrix
function [retf] = lnpdfmvt2(y,mu,cP,nu)

n = rows(y);
a = (n+nu)/2;
lnc = gammaln(a) - gammaln(nu/2) - (n/2)*log(pi*nu);

if cP == 1;         %/* Pinv is the identity matrix */
        e = (y-mu);          %  /* standard normals: k times m matrix */
        ePe = sum(e .* e);   %  /*  m*1 vector  */
        z = lnc - a*log( 1 + ePe/nu );   %/* m*1 vector */
else             %/* Pinv is a full covariance matrix */
        C = cP;       %   the matrix that makes the y uncorrelated */
        e = C*(y-mu);         %   standard normals: k times m matrix */
        ePe = sum(e .* e);    %   m*1 vector  */
        z = lnc + .5*lndet1(C) - a*log( 1 + ePe/nu );   %/* m*1 vector */
end

retf = z;

end