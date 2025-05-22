% /* parameters for IG from mu and std */
% /* e : mu */
% /* s : std */
function [alpha, beta] = paramig(e,s)

d = e.*((e.*e)./(s.*s) + 1);
nu = d./e + 1;
alpha = nu;
beta = d;

end