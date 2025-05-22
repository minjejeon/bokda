% Cholesky
% err = 1 if error, and 0 if no error
% retf = upper triangular
function [retf,err] = chol1(x)

err = 0;
[retf,p] = chol(x);
if p > 0;
    err = 1;
    disp(x);
end
end
