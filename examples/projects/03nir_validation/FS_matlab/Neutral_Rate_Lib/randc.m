% random number generators from chi-square dist with nu degree of freemdom
function [retf] = randc(nu,n)

x = randn(nu,n);  % nu by n
retf = sumc(x.*x); % n by 1

end