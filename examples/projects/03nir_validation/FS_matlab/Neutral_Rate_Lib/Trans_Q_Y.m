function Y = Trans_Q_Y(Q)

[T, N] = size(Q);

Ty = floor(T/4);
Y = zeros(Ty, N);
for t = 1:Ty

    Y(t, :) = meanc(Q(4*t-3:4*t, :))';
    
end

end