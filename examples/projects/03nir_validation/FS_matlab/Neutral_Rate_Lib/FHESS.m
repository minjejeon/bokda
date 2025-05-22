function H = FHESS(f,x)
% FHESS Computes numerical Hessian of a function
% USAGE:
%   H=FHESS(f,x[,additional parameters])
% INPUTS
%   f :  a string name of a function
%   x :  a vector at which to evaluate the Hessian of the function

 k = size(x,1);
 fx = feval(f,x);

 h = eps.^(1/3)*max(abs(x),1e-2);
 xh = x+h;
 h = xh-x;
 ee = sparse(1:k,1:k,h,k,k);

 g = zeros(k,1);
 for i=1:k
   g(i) = feval(f,x+ee(:,i));
 end

 H=h*h';

 for i=1:k
   for j=i:k
     H(i,j) = (feval(f,x+ee(:,i)+ee(:,j))-g(i)-g(j)+fx) ...
                  / H(i,j);
     H(j,i)=H(i,j);
   end
 end