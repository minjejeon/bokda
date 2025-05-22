function [trend, cycle, lgdp] = BN_decomp(Data, p)

lgdp = Data;   
 
T = rows(Data);
%%  Differencing log(GDP)
GDP_growth = lgdp(2:T)-lgdp(1:T-1); % @ growth of GDP (%)= (log(GNP(t)) - log(GNP(t-1)))*100 @

GDP_growth = demeanc(GDP_growth);

% /*==================== Estimating Phi_hat =======================*/
dif_y0 = GDP_growth(p+1:end);

dif_ylag = [ ];
for i = 1:p
   dif_ylag = [dif_ylag, GDP_growth(p-i+1:end-i)]; 
end

Phi=inv(dif_ylag'*dif_ylag)*dif_ylag'*dif_y0; % @ Phi_hat, 4 by 1 @

% /*================ Construction of F matrix =======================*/
F = [Phi';eye(p-1), zeros(p-1,1)]; %   @ p by p @


% /*============ Cyclical Component Deriving ========================*/
T = rows(GDP_growth);
gap = zeros(T-p,1);        % @  t-4 by 1 @
for t = 1:T-p
  gaptmp = F*inv(eye(p)-F)*GDP_growth(t+1:t+p);  %  @ p by 1,  dif_y[itr+1:itr+p]=y_til @
  gap(t) = gaptmp(1,1);
end

gap = [zeros(p+1, 1); gap];
trend = lgdp + gap;   % @ Trend of Y , t-4 by 1 @
cycle = - gap;              % @ Cycle of Y  @

end