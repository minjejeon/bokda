clc;
load path
addpath(path)

load Post_Um
load rhom
load Spec
load Rstm
load Eyestm
load Phim
load Rst_US

Debt = readmatrix('Government_Debt', 'sheet', 'AUS', 'Range', 'B12:B33'); % 거시자료
Debt = Trans_Y_Q(Debt);

Datafile = 'NIR_Data4';   % 중립금리 추정에 사용한 데이타 엑셀파일
Datasheet = 'CA';      % Sheet 명
CA = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'E6:E99'); % 거시자료

CA_AUS = CA;
save CA_AUS CA_AUS

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
log_GD1 = MA(log_GD(1:end-2), 1);
dlog_GD = (log_GD(5:end) - log_GD(1:end-4));
dlog_GD1 = MA(dlog_GD, 1);
T0 = rows(dlog_GD1);
Z = Trend(end-T0+1:end, 4);
Pop1 = MA(Pop(end-T0:end-1, 1), 1);
CA1 = CA(end-T0:end-1, 1);
dCA1 = CA(end-T0+1:end, 1) - CA(end-T0:end-1, 1);
Rst_US1 = Rst_US(end-T0+1:end, 1);
dRst_US1 = Rst_US(end-T0+1:end, 1) - Rst_US(end-T0:end-1, 1);
Xz = [ones(T0, 1), MA(log(Pop1), 4), dlog_GD1, CA1, Rst_US1];
[bhat, sig2hat, stde, t_val, Yhat, ehat] = OLSout(Z, Xz, 1);

Table_Debt = zeros(8, 1);
Table_Debt(1) = bhat(2);
Table_Debt(3) = bhat(3);
Table_Debt(5) = bhat(4);
Table_Debt(7) = bhat(5);
Table_Debt(2) = stde(2);
Table_Debt(4) = stde(3);
Table_Debt(6) = stde(4);
Table_Debt(8) = stde(5);

figure
plot(ehat);
title('residuals')
%%
if Spec.No_FinCycle == 1
    
    Trend_AUS_NF = Trend;
    Cycle_AUS_NF = Cycle;
    Rst_AUS_NF = Rst;
    Eyest_AUS_NF = Eyest;
    rhom_AUS_NF = rhom;
    
    save Trend_AUS_NF Trend_AUS_NF
    save Cycle_AUS_NF Cycle_AUS_NF
    save Rst_AUS_NF Rst_AUS_NF
    save Eyest_AUS_NF Eyest_AUS_NF
    save rhom_AUS_NF rhom_AUS_NF
else
    Ym_AUS = Ym;
    Trend_AUS = Trend;
    Cycle_AUS = Cycle;
    Rst_AUS = Rst;
    Eyest_AUS = Eyest;
    rhom_AUS = rhom;
    
    save Ym_AUS Ym_AUS
    save Trend_AUS Trend_AUS
    save Cycle_AUS Cycle_AUS
    save Rst_AUS Rst_AUS
    save Eyest_AUS Eyest_AUS
    save rhom_AUS rhom_AUS
end



