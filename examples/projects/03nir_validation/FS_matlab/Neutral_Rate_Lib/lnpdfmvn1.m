function [z] = lnpdfmvn1(y,mu,P)	% uses precision instead of var */

C = cholmod(P);            % the matrix that makes the y uncorrelated */
e = C*(y-mu);            % standard normals: k times m matrix */
z = 0.5*lndet1(C) + sumc(lnpdfn1(e));   % the log of the density */

end