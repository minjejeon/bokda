% n = MCMC size
% k = # of variables to be predicted
% yfm = predictive values, n by k
% yf = actual values, k by 1
function [ppc,D,W,ppcm,Dm,Wm] = PPC(yfm,yf)

k = 1000;
n = rows(yf);
Dm = var(yfm)';  % k by 1
D = sumc(Dm);

Wm = zeros(n,1);
for i = 1:n
    pred_err = meanc(yfm(:,i)) - yf(i); % 1 by 1
    Wm(i) = (pred_err^2)*k/(k+1); % 1 by 1
end
W = sumc(Wm);
ppc = sqrt(D + W);
ppcm = sqrt(Dm + Wm);
end
