function autocrm = autocorr(x, maxlag)

if cols(x) > 1 
  error('x must be a vector');
  return
end

n = rows(x);

if n < maxlag 
  error('rows(x) must be bigger than lags');
  return
end

maxlag1 = maxlag + 1;
autocrm = ones(maxlag1,1);

for i = 1:maxlag;
  ind1 = x(i+1:n);
  ind2 = x(1:n-i);
  tem = corrcoef(ind1, ind2);
  autocrm(i+1) = tem(1,2);
end