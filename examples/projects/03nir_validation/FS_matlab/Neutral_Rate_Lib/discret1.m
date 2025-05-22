function [retf] = discret1(p,n)

psum = cumsumc(p);
psuml = [0;psum(1:rows(p)-1)];
store = zeros(n,1);

i = 1;
while i <= n
    u = rand(1)*ones(cols(p),1);
    ind = gt(u,psuml); 
    ind1 = le(u,psum);
    iseql = ind == ind1;
    [~,maxind] = max(iseql);    
    store(i) = maxind;
    i = i + 1;
end

retf = round(store);
end
