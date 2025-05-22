function pplot(x, Y_Left, Y_Right)

if nargin < 3  % x축 없이 자료만 넣을 경우
    Y_Right = Y_Left;
    Y_Left = x;
    x = 1:rows(Y_Left);
end

plot(x, Y_Left);

yyaxis right
plot(x, Y_Right);


end