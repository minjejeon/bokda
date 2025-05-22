function Data = MA_symetric2(Data0, MA_order)

Data = Data0;

T = rows(Data);
if MA_order > 0
    
    for t = 1:MA_order
        Data(t, :) = meanc(Data0(1:MA_order, :))';
    end
    
    for t = (MA_order+1):(T-MA_order)
        Data(t, :) = meanc(Data0(t-MA_order:t+MA_order, :))';
    end
    
    for t = T-MA_order+1:T
        Data(t, :) = meanc(Data0(T- MA_order + 1:T, :))';
    end
    
    
end
end