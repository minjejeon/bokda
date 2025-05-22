function counter_iter(iter, freq, n)

if nargin < 2
    freq = 100;
end

    [~, resid] = minresid(iter,freq);
        if resid == 0
             clc
            disp(['Current iteration is ',num2str(iter)]);
            disp(['MCMC size = ', num2str(n)])
            disp(['About ', num2str(round(100*(iter/n))),'% is done.'])
        end
        
end