function Y = diffc(X, Lag)
% X는 T by Lag 행렬
% Y는 (T-Lag) by K 증가율, 단위는 퍼센트

isX_positive = X > 0;

if minc(isX_positive) == 0
    disp('error: INPUT이 0 또는 음수인 entry를 포함')
    return
end

lnX = log(X);
Y = 100*(lnX(Lag+1:end, :) - lnX(1:end-Lag, :));

end