function Q = Trans_M_Q(M)

[T, N] = size(M);
Q = zeros(T/3, N);
for t = 3:3:T
    
    Q(t/3, :) = meanc(M(t-2:t,:))';
end

end