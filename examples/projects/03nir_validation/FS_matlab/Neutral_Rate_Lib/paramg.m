% /* parameters for Gamma from mu and std */
% /* e : mu */
% /* s : std */
% /* mu = a/b, s^2 = a/(b^2) */
function [retf1, retf2] = paramg(e,s)

s2 = s.*s;
b = e./s2;
a = e.*b;

retf1 = a;
retf2 = b;

end
