% unconditional probability
% tranm = transition matrix
function [uncdprob] = uncondprob(tranm)

s = rows(tranm);
A = [(eye(s)-tranm);ones(1,s)]; %  s+1 by s 
EN = [zeros(s,1);1];          %  (s+1) by 1
AA = A'*A;
AAA = AA\A';     %  s by s+1
uncdprob = AAA*EN;     %  Unconditional Probability,  s by 1

end

