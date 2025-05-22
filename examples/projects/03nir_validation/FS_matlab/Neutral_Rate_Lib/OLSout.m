%%% INPUT
% Y :  dependent variable
% X :  independent variable

%%% OUTPUT
% bhat : k by 1, Estimates for b1 and b2
% Yhat : T by 1, fitted value
% ehat : T by 1, residuals
% sig2hat : Estimates of variance
% varbhat : k by k, variance of bhat
% stde : k by 1, standard error of bhat
% t_val : k by 1, t values
% mY : sample mean of the dependent variable
% TSS : TSS
% RSS : RSS
% R2 : R-square
% R2_ : Adjusted R-square
% SC : SC
% AIC : AIC

function [bhat, sig2hat, stde, t_val, Yhat, ehat, varbhat, mY, TSS, RSS, R2, R2_, SC, AIC, p_val] = OLSout(Y,X,printi)

T = rows(Y);
k = cols(X);
X2 = X'*X;
X2 = 0.5*(X2 + X2');
XY = X'*Y;

bhat = X2\XY; % k by 1, Estimates for b1 and b2
Yhat = X*bhat; % T by 1, fitted value
ehat = Y - Yhat; % T by 1, residuals
sig2hat = ehat'*ehat/(T-k); % Estimates of variance
X2inv = invpd(X2);
varbhat = sig2hat*X2inv; % k by k, variance of bhat
stde = sqrt(diag(varbhat)); % k by 1, standard error
b0 = zeros(k,1);  % null hypothesis
t_val = (bhat - b0)./ stde; % k by 1, t values
p_val = 2*(1-cdfn(abs(t_val))); % k by 1, t values


mY = meanc(Y); % sample mean of the dependent variable
TSS = Y'*Y - T*mY^2; % TSS
RSS = ehat'*ehat;  % RSS
R2 = 1 - RSS/TSS;  % R2
R2_ = 1 - (T-1)*RSS/(TSS*(T-k)); % Adjusted R2
SC = log(RSS/T) - k/T*log(T); % SC
AIC = log(RSS/T) - 2*k/T;  % AIC

if printi > 0
% Results % 
disp('==================================================================');
disp(['  Estimates  ',   'S.E.      ',' t value  ','p value']);
disp('------------------------------------------------------------------');
disp([bhat stde t_val p_val]);
disp('------------------------------------------------------------------');
disp(['S.E. of regression is   ', num2str(sqrt(sig2hat))]);
disp(['R2 is   ', num2str(R2)]);
disp(['adjusted R2 is   ', num2str(R2_)]);
disp(['SC is   ', num2str(SC)]);
disp(['AIC is   ', num2str(AIC)]);
disp('------------------------------------------------------------------');

if printi == 2
    
    subplot(2,1,1); 
    plot([Y Yhat]);
    title('Actual and Fitted')
    subplot(2,1,2); 
    plot([ehat zeros(T,1)])
    title('Residual')
    
end

end

end


