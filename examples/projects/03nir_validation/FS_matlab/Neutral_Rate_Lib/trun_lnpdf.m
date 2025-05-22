% y, a, mu and Sig2 are all vectors
function [trun_lnpdfy] = trun_lnpdf(y,mu,Sig2,a)
   
   lnpdfy = lnpdfn(y,mu,Sig2);
   cdfy = normcdf(a,mu,Sig2);
   
   cdfy = minc([cdfy;0.9999999]);
      
   trun_lnpdfy = lnpdfy - log(1-cdfy);
   trun_lnpdfy = sumc(trun_lnpdfy);
   
end