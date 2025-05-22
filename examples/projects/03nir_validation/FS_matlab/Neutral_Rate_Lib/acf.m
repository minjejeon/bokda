% computing autocorrelation function 
% /* maxac: maximum lag of autocorrelations to compute */
function [retf] = acf(x,maxac)

retf = autocorr(x,maxac);
retf = retf(2:maxac+1);

end