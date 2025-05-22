function y = lnpdfDirich(X, a)
% Compute log pdf of a Dirichlet distribution.
%   X: d x 1 data matrix satifying (sum(X,1)==ones(1,n) && X>=0)
%   a: d x 1 parameters
%   y: 1 x 1 probability density
% Written by Mo Chen (mochen80@gmail.com).
X = bsxfun(@times,X,1./sum(X,1));
if size(a,1) == 1
    a = repmat(a,size(X,1),1);
end
c = gammaln(sum(a,1))-sum(gammaln(a),1);
g = (a-1)'*log(X);
y = bsxfun(@plus,g,c');