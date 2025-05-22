function Q = Trans_Y_Q(Y)

[T, N] = size(Y);
Q = zeros(4*(T-1), N);
for t = 1:(T-1)
    int = (Y(t+1)-Y(t))/4;
    if abs(int) > 0
        q = (Y(t)+int):int:Y(t+1);
    else
        q = Y(t+1)*ones(4,1);
    end
    Q(4*(t-1)+1:4*t, :) = q';
end

end