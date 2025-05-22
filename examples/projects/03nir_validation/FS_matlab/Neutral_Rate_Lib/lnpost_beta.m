function lnpdf_beta = lnpost_beta(Y0,YLm, beta_st, p, b_, var_, Omega_inv, NonZeroRestriction)

NonZeroRestriction = NonZeroRestriction == 1;
T0 = rows(Y0); % = T-p
k = cols(Y0);
X = YLm; % 설명변수, 3차원
XX = zeros(p*k^2);
XY = zeros(p*k^2,1);

for t = 1:T0
    Xt = X(:,:,t);
    XX = XX + Xt'*Omega_inv*Xt;
    XY = XY + Xt'*Omega_inv*Y0(t,:)';
end

precb_ = invpd(var_);
B1_inv = precb_ + XX;
B1_inv = 0.5*(B1_inv + B1_inv');
B1 = invpd(B1_inv);
B1 = 0.5*(B1 + B1');
A = XY + precb_*b_; % b_ = B0
BA = B1*A; % full conditional mean

BA = BA(NonZeroRestriction==1);
B1_inv = B1_inv(NonZeroRestriction==1, NonZeroRestriction==1);

lnpdf_beta = lnpdfmvn1(beta_st(NonZeroRestriction==1), BA, B1_inv); % beta sampling 하기


end
