% random number generators from chi-square dist with nu degree of freemdom
function [retf] = randb(p,n)

tmp = rand(n,1);  % nu by n
retf = tmp < p; % n by 1

end