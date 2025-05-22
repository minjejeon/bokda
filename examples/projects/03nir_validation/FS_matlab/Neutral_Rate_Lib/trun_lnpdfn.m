% y, a, b, mu and Sig2 are all vectors
% log density of truncated normal over [a, b]
function [trun_lnpdfy] = trun_lnpdfn(y,mu,Sig2,a,b)
   
   lnpdfy = lnpdfn(y,mu,Sig2);
   low_cdfy = normcdf(a,mu,Sig2);
   upp_cdfy = 1 - normcdf(b,mu,Sig2);
   cdfy = low_cdfy + upp_cdfy;
   cdfy = minc([cdfy;0.9999999]);
      
   trun_lnpdfy = lnpdfy - log(1-cdfy);
   trun_lnpdfy = sumc(trun_lnpdfy);
   
end