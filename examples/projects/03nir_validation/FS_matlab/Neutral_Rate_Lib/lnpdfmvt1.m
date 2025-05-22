% log of multivariate t density 
function [retf] = lnpdfmvt1(y,mu,P,nu)

n = rows(y);
a = (n+nu)/2;
lnc = gammaln(a) - gammaln(nu/2) - (n/2)*log(pi*nu);

if P == 1;         %/* Pinv is the identity matrix */
        e = (y-mu);          %  /* standard normals: k times m matrix */
        ePe = sum(e .* e);   %  /*  m*1 vector  */
        z = lnc - a*log( 1 + ePe/nu );   %/* m*1 vector */
else             %/* Pinv is a full covariance matrix */
        C = cholmod(P);       %   the matrix that makes the y uncorrelated */
        e = C*(y-mu);         %   standard normals: k times m matrix */
        ePe = sum(e .* e);    %   m*1 vector  */
        z = lnc + .5*lndet1(C) - a*log( 1 + ePe/nu );   %/* m*1 vector */
end
retf = z;

end