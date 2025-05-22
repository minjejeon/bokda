function [mini, r, c, z] =  findmin(X)

mini = min(X(:));
[r, c, z] = ind2sub(size(X),find(X==mini));

end