% log of multiple univariate t density 
function [z] = lnpdft(y,mu,sig2,nu)

n = rows(y);
a = (nu+1)/2;
lnc = gammaln(a) - gammaln(nu/2) - 0.5*log(pi*nu);

P = ones(n,1)./sig2;
C = sqrt(P);       %   the matrix that makes the y uncorrelated */
e = C.*(y-mu);         %   standard normals: k times m matrix */
ePe = e .* e;    %   m*1 vector  */
z = lnc + log(C) - a*log( 1 + ePe/nu );   %/* m*1 vector */


end