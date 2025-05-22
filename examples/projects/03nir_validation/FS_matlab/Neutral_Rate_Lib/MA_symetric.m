function [Trend, Cycle] = MA_symetric(Data, MA_order)

Trend = Data;
T = rows(Data);
if MA_order > 0
    
    for t = 1:MA_order
        Trend(t, :) = meanc(Data(1:t+MA_order, :))';
    end
    
    for t = (MA_order+1):(T-MA_order)
        Trend(t, :) = meanc(Data(t-MA_order:t+MA_order, :))';
    end
    
    for t = T-MA_order+1:T
        Trend(t, :) = meanc(Data(t-MA_order:T, :))';
    end
        
end

Cycle = Data - Trend;

end