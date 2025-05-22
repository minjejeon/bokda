function Omega = Gen_Omega(Trend, a_0, d_0 )

[T, M] = size(Trend);
Omega = eye(M);


for m = 1:M
   e = Trend(2:end, m) - Trend(1:end-1, m);
   a_1 = a_0(m) + T;
   d_1 = d_0(m) + e'*e;
   Omega(m, m) = randig(a_1/2, d_1/2, 1, 1); % sig2 sampling
end

end
