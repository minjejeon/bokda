% Truncated multivariate normal density
% mu = k by 1 mean
% Sigma = k by k matrix
% xl = k by 1 lower bound
% xu = k by 1upper bound

function y = lnpdftmvn(X,mu,Sigma,xl,xu)

   normalconst = mvncdf(xl,xu,mu,Sigma); % normalizing constant
   y = lnpdfmvn(X,mu,Sigma) - log(normalconst);

end