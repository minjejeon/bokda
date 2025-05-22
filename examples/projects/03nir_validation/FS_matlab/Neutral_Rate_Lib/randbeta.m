% mean = a/(a+b)
% x = m by n matrix
function x = randbeta(m,n,a,b)

    
    mn = m*n;
    if mn == 1
        
        A = rand(a,1);
        B = rand(b,1);
        A = sumc(log(A));
        B = sumc(log(B));
        x = A/(A+B);
        
    else
        
        A = rand(a,mn);
        B = rand(b,mn);
        A = sumc(log(A)); % mn by 1
        B = sumc(log(B)); % mn by 1
        x = A./(A+B);  % mn by 1
        
        if n > 1
            x = reshape(x,m,n);
        end
    end


end