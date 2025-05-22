function [stm] = makeStm(sm,ns)

n = rows(sm);
stm = zeros(n,ns);
for t = 1:n
   st = sm(t);
   stm(t,st) = 1;
end

end