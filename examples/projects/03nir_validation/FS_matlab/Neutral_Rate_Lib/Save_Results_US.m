clc;
load path 
addpath(path)

load Post_Um 
load rhom
load Spec 
load Rstm 
load Eyestm 
load Phim

Debt = readmatrix('Government_Debt', 'sheet', 'US', 'Range', 'B12:B33'); % 거시자료
Debt = Trans_Y_Q(Debt);

Datafile = 'NIR_Data4_per_capita';   % 중립금리 추정에 사용한 데이타 엑셀파일
Datasheet = 'CA';      % Sheet 명
CA = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'D6:D99'); % 거시자료

CA_US = CA;
save CA_US CA_US


Pop = Spec.Pop;

[n1, T, K] = size(Post_Um);

Ym = Spec.Ym;
M = cols(Ym);
Trend = zeros(T, M);
Cycle = Trend;
for m = 1:M
  Trend(:, m) = meanc(Post_Um(:, :, m));
  Cycle(:, m) = meanc(Post_Um(:, :, M+m));
  
  Adj_term = meanc(Cycle(:, m));
  Cycle(:, m) = Cycle(:, m) - Adj_term;
  Trend(:, m) = Trend(:, m) + Adj_term;

end

Rst = meanc(Rstm);
Eyest = meanc(Eyestm);

Eyest = Eyest + Adj_term;
Rst = Rst + Adj_term;
if Spec.No_FinCycle == 1
    Rst_NF = Rst;
    Eyest_NF = Eyest;
    save Eyest_NF Eyest_NF
    save Rst_NF Rst_NF
else
    save Eyest Eyest
    save Rst Rst
end

%%
figure
subplot(2,2,1)
plot([Ym(:, 1), Trend(:, 1)])

subplot(2,2,2)
plot([Ym(:, 2), Trend(:, 2)])

subplot(2,2,3)
plot([Ym(:, 3), Trend(:, 3)])

subplot(2,2,4)
plot([Ym(:, 4), Eyest])
saveas(gcf,'fig_trend.png')

figure
plot([Ym(:, 4), Trend(:, 1:2), Trend(:, 4)])
legend('금리', 'G', 'P', 'Z')
saveas(gcf,'fig_X.png')

figure
plot([Ym(:, 4),Eyest, Rst])
legend('금리', 'Eyest', 'Rst')
saveas(gcf,'fig_Rstar.png')

figure
subplot(2,2,1)
plot(Cycle(:, 1))
grid on

subplot(2,2,2)
plot(Cycle(:, 2))
grid on

subplot(2,2,3)
plot(Cycle(:, 3))
grid on

subplot(2,2,4)
plot(Cycle(:, 4))
grid on

sgtitle('Cycles')
saveas(gcf,'fig_cycle.png')
%%
figure
subplot(1,2,1)
histogram(rhom(:, 1))

subplot(1,2,2)
histogram(rhom(:, 2))
%%
Table_Rho = zeros(4, 1);
Table_Rho(1) = meanc(rhom(:, 1));
Table_Rho(3) = meanc(rhom(:, 2));

Table_Rho(2) = stdc(rhom(:, 1));
Table_Rho(4) = stdc(rhom(:, 2));

Table_Phi = zeros(2*M, M);
Phi_hat = meanc(Phim);
Phi_se = stdc(Phim);
Table_Phi([1,3,5,7], :) = reshape(Phi_hat, M, M);
Table_Phi([2,4,6,8], :) = reshape(-Phi_se, M, M);


%%
log_GD = log(Debt);
log_GD1 = MA(log_GD(1:end-6), 1);
dlog_GD1 = (log_GD1(5:end) - log_GD1(1:end-4));
T0 = rows(dlog_GD1);
Z = Trend(end-T0+1:end, 4);
Pop1 = Pop(end-T0+1:end, 1);
CA1 = MA(CA(end-T0-1:end-2, 1), 2);
dCA1 = MA(CA(end-T0-1:end-2, 1), 2) - MA(CA(end-T0-2:end-3, 1), 2);
Xz = [ones(T0, 1), MA(log(Pop1), 1), dlog_GD1, dCA1];
[bhat, sig2hat, stde, t_val, Yhat, ehat] = OLSout(Z, Xz, 1);

Table_Debt = zeros(6, 1);
Table_Debt(1) = bhat(2);
Table_Debt(3) = bhat(3);
Table_Debt(5) = bhat(4);
Table_Debt(2) = stde(2);
Table_Debt(4) = stde(3);
Table_Debt(6) = stde(4);


figure
plot(ehat);
title('residuals')
%%


%%
if Spec.No_FinCycle == 0
    
Trend_US_NF = Trend;
Cycle_US_NF = Cycle;
Rst_US_NF = Rst;
Eyest_US_NF = Eyest;
rhom_US_NF = rhom;

save Trend_US_NF Trend_US_NF
save Cycle_US_NF Cycle_US_NF
save Rst_US_NF Rst_US_NF
save Eyest_US_NF Eyest_US_NF
save rhom_US_NF rhom_US_NF

else
  Ym_US = Ym;
Trend_US = Trend;
Cycle_US = Cycle;
Rst_US = Rst;
Eyest_US = Eyest;
rhom_US = rhom;

save Ym_US Ym_US
save Trend_US Trend_US
save Cycle_US Cycle_US
save Rst_US Rst_US
save Eyest_US Eyest_US
save rhom_US rhom_US

end

