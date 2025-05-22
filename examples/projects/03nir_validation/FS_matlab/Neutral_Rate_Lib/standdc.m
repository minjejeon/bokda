function y = standdc(x)

x = demeanc(x); % T by k
sd = stdc(x); % k by 1
sd = kron(sd', ones(rows(x), 1)); % T by k
y = x ./ sd;


end