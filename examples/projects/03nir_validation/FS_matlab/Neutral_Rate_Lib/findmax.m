function [maxi, r, c, z] =  findmax(X)

maxi = max(X(:));
[r, c, z] = ind2sub(size(X),find(X==maxi));

end