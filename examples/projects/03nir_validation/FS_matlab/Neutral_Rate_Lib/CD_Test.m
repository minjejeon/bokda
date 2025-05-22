%  This proc computes Geweke's convergence test statistics 
%  MHmat = n by q
% q = number of parameters
% n1 = MCMC size
function [CDm] = CD_Test(MHmat)

n = rows(MHmat);
T1 = 0.2*n;
T2 =  0.2*n;
T = T1 + T2;
MHm1 = MHmat(1:T1,:);
MHm2 = MHmat(T-T2+1:T,:);
d1 = meanc(MHm1);
d2 = meanc(MHm2);
v1 = var(MHm1)'; % q by 1 
v2 = var(MHm2)'; % q by 1 

stde = sqrt(v1/T1 + v2/T2);
CDm = (d1 - d2)./stde;

end