function H = FHESSnew(f,x,index)
% FHESS Computes numerical Hessian of a function
% USAGE:
%   H=FHESS(f,x[,additional parameters])
% INPUTS
%   f :  a string name of a function
%   x :  a vector at which to evaluate the Hessian of the function


 k = size(index,1);
 fx = feval(f,x);
 xarg = x(index);
 
 h = eps.^(1/5)*max(abs(xarg),1e-2);
 xargh = x(index) + h;
 h = xargh - x(index);
 disp(h);
 disp(k);
 ee = sparse(1:k,1:k,h,k,k);

 g = zeros(k,1);
 for i=1:k
   xee = x;
   xee(index) = xarg + ee(:,i);  
   g(i) = feval(f,xee);
 end

 H=h*h';

 for i=1:k
   for j=i:k
       xeeij  = x;
       xargeeij = xarg + ee(:,i) + ee(:,j);
       xeeij(index) = xargeeij;
     H(i,j) = (feval(f,xeeij)-g(i)-g(j)+fx) ...
                  / H(i,j);
     H(j,i)=H(i,j);
   end
 end