% Note that
% Suppose that x = randig(alpha,beta,1,1)
% E(x) = beta/(alpha-1)
% Var(x) = beta^2/[(alpha-2)*(alpha-1)^2]

function [ig] = randig(alpha,beta,m,n)

if nargin < 3
    m = 1;
    n = 1;
end
gam = randgam(alpha,beta,m,n);
ig = ones(m,n)./gam;

end