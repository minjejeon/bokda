clc;
load path
addpath(path)

load Post_Um
load rhom
load Spec
load Rstm
load Eyestm
load Phim

Debt = readmatrix('Government_Debt', 'sheet', 'KOR', 'Range', 'B12:B33'); % 거시자료
Debt = Trans_Y_Q(Debt);

Datafile = 'NIR_Data4';   % 중립금리 추정에 사용한 데이타 엑셀파일
Datasheet = 'CA';      % Sheet 명
CA = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'B6:B99'); % 거시자료

CA_KOR = CA;
save CA_KOR CA_KOR


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

figure
subplot(1,2,1)
histogram(rhom(:, 1))

subplot(1,2,2)
histogram(rhom(:, 2))

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
Z = Trend(1:end, 4);
T = rows(Z);
log_GD = log(Debt);
log_GD1 = log_GD(end-T+1:end);
CA1 = MA(CA(end-T+1:end, 1), 2);
Pop1 = Pop(end-T+1:end, 1);
dCA1 = MA(CA(end-T+1:end, 1), 2) - MA(CA(end-T:end-1, 1), 2);
dRst_US1 = Rst_US(end-T+1:end, 1) - Rst_US(end-T:end-1, 1);
Xz = [ones(T, 1), MA(log(Pop1), 4), log_GD1, dCA1, dRst_US1];

[bhat, sig2hat, stde, t_val, Yhat, ehat] = OLSout(Z, Xz, 1);

Table_Debt1 = zeros(8, 1);
Table_Debt1(1) = bhat(2);
Table_Debt1(3) = bhat(3);
Table_Debt1(5) = bhat(4);
Table_Debt1(7) = bhat(5);
Table_Debt1(2) = stde(2);
Table_Debt1(4) = stde(3);
Table_Debt1(6) = stde(4);
Table_Debt1(8) = stde(5);

figure
plot(ehat);
title('residuals')

%%
T0 = rows(Pop);
dlog_GD = (log_GD(5:end) - log_GD(1:end-4));
CA2 = MA(CA(end-T0+1:end, 1), 2);
Rst_US2 = Rst_US(end-T0+1:end, 1);

Z = Trend(:, 4);
Xz = [ones(T0, 1), MA(log(Pop), 4), dlog_GD(end-T0+1:end, 1), CA2, Rst_US2];
[bhat, sig2hat, stde, t_val, Yhat, ehat] = OLSout(Z, Xz, 1);

Table_Debt2 = zeros(6, 1);
Table_Debt2(1) = bhat(2);
Table_Debt2(3) = bhat(3);
Table_Debt2(5) = bhat(4);
Table_Debt2(2) = stde(2);
Table_Debt2(4) = stde(3);
Table_Debt2(6) = stde(4);
Table_Debt2(6) = stde(4);
Table_Debt2(8) = stde(5);
figure
plot(ehat);
title('residuals')

%%
if Spec.No_FinCycle == 1
    
    
    Trend_KOR_NF = Trend;
    Cycle_KOR_NF = Cycle;
    Rst_KOR_NF = Rst;
    Eyest_KOR_NF = Eyest;
    rhom_KOR_NF = rhom;
    
    save Trend_KOR_NF Trend_KOR_NF
    save Cycle_KOR_NF Cycle_KOR_NF
    save Rst_KOR_NF Rst_KOR_NF
    save Eyest_KOR_NF Eyest_KOR_NF
    save rhom_KOR_NF rhom_KOR_NF
else
    Ym_KOR = Ym;
    Trend_KOR = Trend;
    Cycle_KOR = Cycle;
    Rst_KOR = Rst;
    Eyest_KOR = Eyest;
    rhom_KOR = rhom;
    
    save Ym_KOR Ym_KOR
    save Trend_KOR Trend_KOR
    save Cycle_KOR Cycle_KOR
    save Rst_KOR Rst_KOR
    save Eyest_KOR Eyest_KOR
    save rhom_KOR rhom_KOR
    
end

