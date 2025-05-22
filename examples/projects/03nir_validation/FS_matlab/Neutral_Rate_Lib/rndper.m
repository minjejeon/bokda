% /* proc to randomly permute sequence of numbers */
function [retf] = rndper(y)

n = length(y);
z = y;

k = n;
while k > 1
i = ceil(rand(1,1)*k);
zi = z(i,:);
zk = z(k,:);
    z(i,:) = zk;   %/* interchange values */
    z(k,:) = zi;
k = k - 1;
end

retf = z;
end
