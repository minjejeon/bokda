function [Xm, trendm, bm, tm]  = detrend(Y)

[T, M] = size(Y);
x = 1:T;
x = [ones(T, 1), x'];
trendm = Y;
bm = zeros(2, M);
tm = zeros(2, M);
for i = 1:M
    [bhat, ~, ~, t_val, Yhat] = OLSout(Y(:, i), x, 0);
    bm(:, i) = bhat;
    tm(:, i) = t_val;
    trendm(:, i) = Yhat;
end

Xm = Y - trendm;

end

