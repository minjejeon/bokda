% y, a, mu = vectors, Sig2 = matrix
function [trun_lnpdfy] = trun_lnpdfmvn(y,mu,Sig2,a)

    lnpdfy = lnpdfmvn(y,mu,Sig2);
   cdfy = mvncdf(a',mu',Sig2);
 %   cdfy = 0;
    cdfy = minc([cdfy;0.9999999]);
      
    trun_lnpdfy = lnpdfy - log(1-cdfy);
   
end