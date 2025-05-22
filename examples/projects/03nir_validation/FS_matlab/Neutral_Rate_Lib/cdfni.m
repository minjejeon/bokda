% Normal inverse cumulative distribution function
% 0<p<1
function [retf] = cdfni(p)

retf = norminv(p,0,1);

end