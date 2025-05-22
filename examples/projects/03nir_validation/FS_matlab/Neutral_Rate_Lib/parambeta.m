% m = mean
% s = standard error
function [a,b] = parambeta(m,s)

V = s.*s;
b = (1-m).*( (1-m)./(m.*V) - 1);
a = (b.*m)./(1-m);

end