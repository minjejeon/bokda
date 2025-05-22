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

Debt = readmatrix('Government_Debt', 'sheet', 'Euro', 'Range', 'B12:B33'); % 거시자료
Debt = Trans_Y_Q(Debt);
Pop = Spec.Pop;

Datafile = 'NIR_Data4';   % 중립금리 추정에 사용한 데이타 엑셀파일
Datasheet = 'CA';      % Sheet 명
CA = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'C6:C99'); % 거시자료

CA_Euro = CA;
save CA_Euro CA_Euro

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
if Spec.No_FinCycle == 0
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
log_GD = log(Debt);
% dlog_GD = (log_GD(2:end) - log_GD(1:end-1));
T0 = rows(log_GD);
Z = Trend(end-T0+1:end, 4);
Pop1 = Pop(end-T0+1:end, 1);
CA1 = MA(CA(end-T0+1:end, 1), 1);
dCA1 = MA(CA(end-T0+1:end, 1), 8) - MA(CA(end-T0:end-1, 1), 8);
Rst_US1 = Rst_US(end-T0-1:end-2, 1);
dRst_US1 = Rst_US(end-T0+1:end, 1) - Rst_US(end-T0:end-1, 1);
Xz = [ones(T0, 1), MA(log(Pop1), 4), log_GD, CA1, Rst_US1];
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



if Spec.No_FinCycle == 0
    
Trend_Euro_NF = Trend;
Cycle_Euro_NF = Cycle;
Rst_Euro_NF = Rst;
Eyest_Euro_NF = Eyest;
rhom_Euro_NF = rhom;

save Trend_Euro_NF Trend_Euro_NF
save Cycle_Euro_NF Cycle_Euro_NF
save Rst_Euro_NF Rst_Euro_NF
save Eyest_Euro_NF Eyest_Euro_NF
save rhom_Euro_NF rhom_Euro_NF
    
else
Ym_Euro = Ym;
Trend_Euro = Trend;
Cycle_Euro = Cycle;
Rst_Euro = Rst;
Eyest_Euro = Eyest;
rhom_Euro = rhom;

save Ym_Euro Ym_Euro
save Trend_Euro Trend_Euro
save Cycle_Euro Cycle_Euro
save Rst_Euro Rst_Euro
save Eyest_Euro Eyest_Euro
save rhom_Euro rhom_Euro
end





