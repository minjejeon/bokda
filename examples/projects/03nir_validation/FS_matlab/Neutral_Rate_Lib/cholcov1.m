% Cholesky-like covariance decomposition
% err = 1 if error, and 0 if no error
% retf = upper triangular
function [retf,err] = cholcov1(x)

err = 0;
[retf,~] = cholcov(x);
if isempty(retf) == 1
    err = 1;
end

end
