function Data = MA(Data, MA_order)

if MA_order > 1
    
    for t = MA_order:rows(Data)
        Data(t, :) = meanc(Data(t-MA_order+1:t, :))';
    end
    
end

end