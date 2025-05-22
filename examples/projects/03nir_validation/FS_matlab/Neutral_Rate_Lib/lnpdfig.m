% to compute the log inverted gamma density on a grid
% beta is also a vector 
% alpha = shape parameter
% beta = scale parameter
% mean = beta / (alpha - 1)
function z = lnpdfig(sig2, alpha, beta)

   c = alpha.*log(beta) - gammaln(alpha);
   z = c - (alpha+1).*log(sig2) - beta./sig2;

end