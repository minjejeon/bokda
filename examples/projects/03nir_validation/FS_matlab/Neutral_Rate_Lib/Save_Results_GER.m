clc;
load path 
addpath(path)

load Post_Um 
load rhom
load Spec 
load Rstm 
load Eyestm 
load Phim

Debt = readmatrix('Government_Debt', 'sheet', 'GER', 'Range', 'B12:B33'); % 거시자료
Debt = Trans_Y_Q(Debt);


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
log_GD = log(Debt);
% dlog_GD = (log_GD(5:end) - log_GD(1:end-4));
T0 = rows(log_GD);
Z = Trend(end-T0+1:end, 4);
Pop1 = Pop(end-T0+1:end, 1);
Xz = [ones(T0, 1), MA(log(Pop1), 4), log_GD];
[bhat, sig2hat, stde, t_val, Yhat, ehat] = OLSout(Z, Xz, 1);

Table_Debt = zeros(4, 1);
Table_Debt(1) = bhat(2);
Table_Debt(3) = bhat(3);
Table_Debt(2) = stde(2);
Table_Debt(4) = stde(3);

figure
plot(ehat);
title('residuals')

%%
Ym_GER = Ym;
Trend_GER = Trend;
Cycle_GER = Cycle;
Rst_GER = Rst;
Eyest_GER = Eyest;
rhom_GER = rhom;

save Ym_GER Ym_GER
save Trend_GER Trend_GER
save Cycle_GER Cycle_GER
save Rst_GER Rst_GER
save Eyest_GER Eyest_GER
save rhom_GER rhom_GER
return



LW = Spec.LW;
%%
q = [0.05, 0.5, 0.95];
[J1, K, n1] = size(Thetam);
J = J1 - 1;

[T, M] = size(Ym);
K = 2*M;

if is_beta_sampling == 1
   disp('beta')
   disp(meanc(betam))
   MHout(betam, 0.05, 50, 1);
end

%% 구조충격에 대한 성장률의 충격반응

titlem = {'temporary demand shock', 'temporary cost shock', 'temporary base rate shock', ...
    'permanent growth shock',  'permanent money supply shock',  'neutral interest rate shock'};

theta_hatm = [ ];
for k = 1:K
    thetam = zeros(n1, J+1);
    for iter = 1:n1
        for j = 1:(J+1)
            thetam(iter, j) = Thetam_Y1m(j, k, iter);
        end
    end
    
    theta_hat = quantile(thetam, q)';
    
    theta_hatm = [theta_hatm, theta_hat];
end


x = 0:J;
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/2 scrsz(3)/3 scrsz(4)/2]);
movegui(f);
for k = 1:K
    subplot(3, 2, k)
    plot(x, theta_hatm(:, 3*k-2), 'b--', x, theta_hatm(:, 3*k-1), 'k-', x, theta_hatm(:, 3*k), 'b--', 'Linewidth', 2)
    hold on
    plot(x, zeros(J+1,1), 'k:')
    title(titlem(k))
    set(gca,'FontSize',12)
end
sgtitle('Responses of GDP growth rate')

saveas(gcf,'fig_IRF_GDP.png')


% 저장하기
horizon = 0:J;
sheet_name = '성장률의충격반응';
writematrix('temporary demand','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('temporary cost','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('temporary base rate','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 
writematrix('permanent growth','Results.xlsx','Sheet', sheet_name, 'Range', 'E1') 
writematrix('permanent money supply','Results.xlsx','Sheet', sheet_name, 'Range', 'F1') 
writematrix('NIR','Results.xlsx','Sheet', sheet_name, 'Range', 'G1') 

writematrix('horizon(분기)','Results.xlsx','Sheet', sheet_name, 'Range', 'A2') 
writematrix(horizon','Results.xlsx','Sheet', sheet_name, 'Range', 'A3:A1000') 

writematrix(theta_hatm,'Results.xlsx','Sheet', sheet_name, 'Range', 'B3:Z1000') 


%% 구조충격에 대한 inflation rate의 충격반응

theta_hatm = [ ];
for k = 1:K
    thetam = zeros(n1, J+1);
    for iter = 1:n1
        for j = 1:(J+1)
            thetam(iter, j) = Thetam_Y2m(j, k, iter);
        end
    end
    
    theta_hat = quantile(thetam, q)';
    
    theta_hatm = [theta_hatm, theta_hat];
end


x = 0:J;
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/2 scrsz(3)/3 scrsz(4)/2]);
movegui(f);
for k = 1:K
    subplot(3, 2, k)
    plot(x, theta_hatm(:, 3*k-2), 'b--', x, theta_hatm(:, 3*k-1), 'k-', x, theta_hatm(:, 3*k), 'b--', 'Linewidth', 2)
    hold on
    plot(x, zeros(J+1,1), 'k:')
    title(titlem(k))
    set(gca,'FontSize',12)
end
sgtitle('Responses of inflation rate')
saveas(gcf,'fig_IRF_Inflation.png')

% 저장하기
horizon = 0:J;
sheet_name = '인플레이션의충격반응';
writematrix('temporary demand','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('temporary cost','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('temporary base rate','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 
writematrix('permanent growth','Results.xlsx','Sheet', sheet_name, 'Range', 'E1') 
writematrix('permanent money supply','Results.xlsx','Sheet', sheet_name, 'Range', 'F1') 
writematrix('NIR','Results.xlsx','Sheet', sheet_name, 'Range', 'G1') 

writematrix('horizon(분기)','Results.xlsx','Sheet', sheet_name, 'Range', 'A2') 
writematrix(horizon','Results.xlsx','Sheet', sheet_name, 'Range', 'A3:A1000') 

writematrix(theta_hatm,'Results.xlsx','Sheet', sheet_name, 'Range', 'B3:Z1000') 

%% 구조충격에 대한 base rate의 충격반응

theta_hatm = [ ];
for k = 1:K
    thetam = zeros(n1, J+1);
    for iter = 1:n1
        for j = 1:(J+1)
            thetam(iter, j) = Thetam_Y3m(j, k, iter);
        end
    end
    
    theta_hat = quantile(thetam, q)';
    
    theta_hatm = [theta_hatm, theta_hat];
end


x = 0:J;
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/2 scrsz(3)/3 scrsz(4)/2]);
movegui(f);
for k = 1:K
    subplot(3, 2, k)
    plot(x, theta_hatm(:, 3*k-2), 'b--', x, theta_hatm(:, 3*k-1), 'k-', x, theta_hatm(:, 3*k), 'b--', 'Linewidth', 2)
    hold on
    plot(x, zeros(J+1,1), 'k:')
    title(titlem(k))
    set(gca,'FontSize',12)
end
sgtitle('Responses of base rate')


% 저장하기
horizon = 0:J;
sheet_name = '기준금리의충격반응';
writematrix('temporary demand','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('temporary cost','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('temporary base rate','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 
writematrix('permanent growth','Results.xlsx','Sheet', sheet_name, 'Range', 'E1') 
writematrix('permanent money supply','Results.xlsx','Sheet', sheet_name, 'Range', 'F1') 
writematrix('NIR','Results.xlsx','Sheet', sheet_name, 'Range', 'G1') 

writematrix('horizon(분기)','Results.xlsx','Sheet', sheet_name, 'Range', 'A2') 
writematrix(horizon','Results.xlsx','Sheet', sheet_name, 'Range', 'A3:A1000') 

writematrix(theta_hatm,'Results.xlsx','Sheet', sheet_name, 'Range', 'B3:Z1000') 


x = 0:J;
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/2 scrsz(3)/3 scrsz(4)/2]);
movegui(f);
indkm = [1;2;4;5];
for k = 1:4
    subplot(2, 2, k)
    indk = indkm(k);
    plot(x, theta_hatm(:, 3*indk-2), 'b--', x, theta_hatm(:, 3*indk-1), 'k-', x, theta_hatm(:, 3*indk), 'b--', 'Linewidth', 2)
    hold on
    plot(x, zeros(J+1,1), 'k:')
    title(titlem(indk))
    set(gca,'FontSize',12)
end
sgtitle('Responses of base rate')
saveas(gcf,'fig_IRF_BaseRate.png')


%%
Y_labelm = {'$\bar{C}(2,1)$'; '$\bar{C}(3,1)$'; '$\bar{C}(1,2)$'; '$\bar{C}(3,2)$'; '$\bar{C}(1,3)$'; '$\bar{C}(2,3)$'; '$\bar{C}(1,4)$'; ...
    '$\bar{C}(2,4)$'; '$\bar{C}(3,4)$'; '$\bar{C}(1,5)$'; '$\bar{C}(2,5)$'; '$\bar{C}(3,5)$'};
indMH = [2;3;4;6;7;8;10;11;12;13;14;15];

MHout_Cbar(Cbarm(:, indMH), 0.05, 50, 1, Y_labelm');

%%
titlem = {'growth cycle', 'inflation cycle', 'interest rate cycle', ...
    'growth trend',  'inflation trend',  'NIR'};
xticklabel = {'03:Q1', '07:Q1', '11:Q1', '15:Q1', '19:Q1', '23:Q1'};
xtick = 5:16:T;
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/1.5 scrsz(3)/2 scrsz(4)/1.5]);
movegui(f);
Um_hat = zeros(T, K);
Trend_hat = zeros(T, M);
for m = 1:M
    subplot(3, 2, 2*m-1)
    Um_k = Post_Um(:, :, m);
    Um_hat(:, m) = meanc(Um_k(1:end, :));
    plot([zeros(T, 1), Um_hat(:, m)]);
    xticks(xtick)
    xticklabels(xticklabel)
    xlim([0, T+1])
    title(titlem(m))
    
    subplot(3, 2, 2*m)
    Um_k = Post_Um(:, :, M+m);
    Um_hat(:, m) = meanc(Um_k(1:end, :));
    plot([zeros(T, 1), Um_hat(:, m)]);
    xlim([0, T+1])
        xticks(xtick)
    xticklabels(xticklabel)
    title(titlem(M+m))
    
    Trend_hat(:, m) = Um_hat(:, m);
end

saveas(gcf,'fig_6_factors.png')


%%
x = 1:T;
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/1.5 scrsz(3)/3 scrsz(4)/1.5]);
movegui(f);
subplot(3, 1, 1)
plot(x,Trend_hat(:, 1), 'linewidth', 2)
xlim([0, T+1])
    xticks(xtick)
    xticklabels(xticklabel)
    ylabel('(%)')
title('(a) growth trend')
set(gca,'FontSize',12)
subplot(3, 1, 2)
plot(x,Trend_hat(:, 2), 'linewidth', 2)
xlim([0, T+1])
    xticks(xtick)
    xticklabels(xticklabel)
      ylabel('(%)')
title('(b) inflatin trend')
set(gca,'FontSize',12)
subplot(3, 1, 3)
plot(x,Trend_hat(:,3), 'linewidth', 2)
xlim([0, T+1])
    xticks(xtick)
    xticklabels(xticklabel)
      ylabel('(%)')
title('(c) neutral interest rate')
set(gca,'FontSize',12)
saveas(gcf,'fig_Permanent_factors.png')

%%
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/1.2 scrsz(3)/3 scrsz(4)/1.2]);
movegui(f);
x = 1:T;
subplot(3, 1, 1)
yyaxis left
plot(x, Ym(:, 1), 'k:', x, Trend_hat(:, 1), 'r-', 'LineWidth', 2)
ylim([-10 10])
yyaxis right
plot(x, Ym(:, 1)-Trend_hat(:, 1), 'b--', x, zeros(T, 1), 'b:', 'LineWidth', 2)
ylim([-7 15])
xlim([0, T+1])
    xticks(xtick)
    xticklabels(xticklabel)
    legend('GDP growth rate (L)', 'trend (L)', 'cycle (R)')
title('(a) GDP growth rate')
set(gca,'FontSize',12)

subplot(3, 1, 2)
yyaxis left
plot(x, Ym(:, 2), 'k:', x, Trend_hat(:, 2), 'r-', 'LineWidth', 2)
ylim([-8 8])
yyaxis right
plot(x, Ym(:, 2)-Trend_hat(:, 2), 'b--', x, zeros(T, 2), 'b:', 'LineWidth', 2)
ylim([-5 10])
xlim([0, T+1])
    xticks(xtick)
    xticklabels(xticklabel)
    legend('inflation rate (L)', 'trend (L)', 'cycle (R)')
title('(b) inflation rate')
set(gca,'FontSize',12)

subplot(3, 1, 3)
yyaxis left
plot(x, Ym(:, 3), 'k:', x, Trend_hat(:, 2) + Trend_hat(:, 3), 'r-', 'LineWidth', 2)
ylim([-5 8])
yyaxis right
plot(x, Ym(:, 3)-Trend_hat(:, 2) - Trend_hat(:, 3), 'b--', x, zeros(T, 2), 'b:', 'LineWidth', 2)
ylim([-2 5])
xlim([0, T+1])
set(gca,'FontSize',12)
    xticks(xtick)
    xticklabels(xticklabel)
    legend('base rate (L)', 'trend (L)', 'cycle (R)')
title('(c) base rate')


saveas(gcf,'fig_trends_cycles.png')

%% 추세와 순환
sheet_name = '추세와순환치';
writematrix('성장률 순환치','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('물가상승률 순환치','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('금리 순환치%','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 
writematrix('성장률추세','Results.xlsx','Sheet', sheet_name, 'Range', 'E1') 
writematrix('물가상승률 추세','Results.xlsx','Sheet', sheet_name, 'Range', 'F1') 
writematrix('금리 추세','Results.xlsx','Sheet', sheet_name, 'Range', 'G1') 

writematrix('Date','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writecell(Spec.date,'Results.xlsx','Sheet', sheet_name, 'Range', 'A2:A1000') 

cycle_hat = Ym - [Trend_hat(:, 1:2), Trend_hat(:, 2)+ Trend_hat(:, 3)];
writematrix(cycle_hat(:, 1),'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:B1000') 
writematrix(cycle_hat(:, 2),'Results.xlsx','Sheet', sheet_name, 'Range', 'C2:C1000') 
writematrix(cycle_hat(:, 3),'Results.xlsx','Sheet', sheet_name, 'Range', 'D2:D1000') 

writematrix(Trend_hat(:, 1),'Results.xlsx','Sheet', sheet_name, 'Range', 'E2:E1000') 
writematrix(Trend_hat(:, 2),'Results.xlsx','Sheet', sheet_name, 'Range', 'F2:F1000') 
writematrix(Trend_hat(:, 2)+ Trend_hat(:, 3),'Results.xlsx','Sheet', sheet_name, 'Range', 'G2:G1000') 

%% 물가상승률 추세 그리기

Trendm = Post_Um(:, :, 5);
Trendm = Trendm(1:end, :);
Trendm = quantile(Trendm, q)';
x = 1:T;
figure
plot(x,Trendm(:, 1), 'r--', x, Trendm(:, 2), 'b-', x, Trendm(:, 3), 'r--', x, zeros(T,1), 'k--')
xlim([0, T+1])
    xticks(xtick)
    xticklabels(xticklabel)
legend('5%', 'Median', '95%')
title('물가상승률 추세')


%% 성장률추세 저장하기
sheet_name = '물가상승률추세';
writematrix('5%','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('Median','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('95%','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 

writematrix('Date','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writecell(Spec.date,'Results.xlsx','Sheet', sheet_name, 'Range', 'A2:A1000') 

writematrix(Trendm,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:D1000') 


%% 성장률 추세 그리기
q = [0.05, 0.5, 0.95];
Trendm = Post_Um(:, :, 4);
Trendm = Trendm(1:end, :);
Trendm = quantile(Trendm, q)';
x = 1:T;
figure
plot(x,Trendm(:, 1), 'r--', x, Trendm(:, 2), 'b-', x, Trendm(:, 3), 'r--', x, zeros(T,1), 'k--')
xlim([0, T+1])
    xticks(xtick)
    xticklabels(xticklabel)
legend('5%', 'Median', '95%')
title('성장률 추세')


%% 성장률추세 저장하기
sheet_name = '성장률추세';
writematrix('5%','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('Median','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('95%','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 

writematrix('Date','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writecell(Spec.date,'Results.xlsx','Sheet', sheet_name, 'Range', 'A2:A1000') 

writematrix(Trendm,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:D1000') 

%% 중립금리 그리기
q = [0.05, 0.5, 0.95];
NIRm = Post_Um(:, :, 6);
NIRm = NIRm(1:end, :);
NIR_hat = quantile(NIRm, q)';
x = 1:T;
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/3 scrsz(3)/3 scrsz(4)/3]);
movegui(f);
plot(x,NIR_hat(:, 1), 'b--', x, NIR_hat(:, 2), 'k-', x, NIR_hat(:, 3), 'b--', 'LineWidth', 2)
hold on
plot(x, zeros(T,1), 'k:')
xlim([0, T+1])
    xticks(xtick)
    xticklabels(xticklabel)
legend('5%', 'Median', '95%')
title('neutral interest rate')
set(gca,'FontSize',14)
disp(['Acceptance rate = ', num2str(Spec.acceptance_rate), '%'])

saveas(gcf,'fig_NIR.png')

% scrsz = get(0,'ScreenSize');
% f=figure('Position',[1 scrsz(4)/3 scrsz(3)/3 scrsz(4)/3]);
% movegui(f);
% plot(x, NIR_hat(:, 1), 'b:', x, NIR_hat(:, 2), 'k-', x, NIR_hat(:, 3), 'b:', x, LW, 'r--', 'LineWidth', 2)
% hold on
% plot(x, zeros(T,1), 'k:')
% xlim([0, T+1])
%     xticks(xtick)
%     xticklabels(xticklabel)
% legend('5%', 'Median', '95%', 'Laubach-Williams')
% title('neutral interest rates')
% set(gca,'FontSize',14)
% 
% saveas(gcf,'fig_LW.png')
%% 중립금리 저장하기
sheet_name = '중립금리';
writematrix('5%','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('Median','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('95%','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 

writematrix('Date','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writecell(Spec.date,'Results.xlsx','Sheet', sheet_name, 'Range', 'A2:A1000') 

writematrix(NIR_hat,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:D1000') 


%%
titlem = {'temporary demand shock', 'temporary cost shock', 'temporary base rate shock', ...
    'permanent growth shock',  'permanent money supply shock',  'neutral interest rate shock'};
[J1, K, n1] = size(Thetam);
J = J1 - 1;
x = 0:J;
scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/2 scrsz(3)/3 scrsz(4)/2]);
movegui(f);
for k = [1,2,4,5]
    thetam = zeros(n1, J+1);
    for iter = 1:n1
        for j = 1:(J+1)
            thetam(iter, j) = Thetam(j, k, iter);
        end
    end
    
    theta_hat = quantile(thetam, q)';
    if k>2
        k = k-1;
    end
    subplot(2, 2, k)
    plot(x,theta_hat(:, 1), 'b--', x, theta_hat(:, 2), 'k-', x, theta_hat(:, 3), 'b--', 'LineWidth',2)
    hold on
    plot(x, zeros(J+1,1), 'k--', 'LineWidth', 1)
   
        if k>2
        title(titlem(k+1))
        else
             title(titlem(k))
        end
    
    set(gca,'FontSize',13)
end
sgtitle('Responses of base rate gap')
saveas(gcf,'fig_IRF.png')

%%
theta_hatm = [ ];
for k = 1:K
    thetam = zeros(n1, J+1);
    for iter = 1:n1
        for j = 1:(J+1)
            thetam(iter, j) = Thetam(j, k, iter);
        end
    end
    
    theta_hat = quantile(thetam, q)';
    
    theta_hatm = [theta_hatm, theta_hat];
end


figure
for k = 1:K
    subplot(3, 2, k)
    plot(x,theta_hatm(:, 3*k-1), x, zeros(J+1,1), 'k--')
    title(titlem(k))
end

%% 중립금리 저장하기
horizon = 0:J;
sheet_name = '기준금리순환치의충격반응';
writematrix('temporary demand','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('temporary cost','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('temporary base rate','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 
writematrix('permanent growth','Results.xlsx','Sheet', sheet_name, 'Range', 'E1') 
writematrix('permanent money supply','Results.xlsx','Sheet', sheet_name, 'Range', 'F1') 
writematrix('NIR','Results.xlsx','Sheet', sheet_name, 'Range', 'G1') 

writematrix('horizon(분기)','Results.xlsx','Sheet', sheet_name, 'Range', 'A2') 
writematrix(horizon','Results.xlsx','Sheet', sheet_name, 'Range', 'A3:A1000') 

writematrix(theta_hatm,'Results.xlsx','Sheet', sheet_name, 'Range', 'B3:Z1000') 


%%
disp('Omega (permant shock variance-covariance)')
K = 2*M;
Omega_hat = reshape(meanc(Omegam), M, M);
Omega_SE = reshape(stdc(Omegam), M, M);
disp(Omega_hat)


disp('permant shock correlation')
disp(corrcov(Omega_hat));
% positive relation betwen permanent growth shock and change in NIR
% negative relation betwen permanent money supply shock and change in NIR

disp('Sigma (transitory shock variances)')
Sigma_hat = meanc(Sigmam);
Sigma_SE = -stdc(Sigmam);
Sigma_table = Sigma_hat;
disp(Sigma_table)

sheet_name = '파라메터(Sigma)';
writematrix('Sigma','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writematrix(Sigma_table,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:C4') 

disp('B')

B_hat = reshape(meanc(Bm), M, M);
B_SE = reshape(stdc(Bm), M, M);

B_table = zeros(K, M);
for m = 1:M
    
    B_table(2*m-1, :) = B_hat(m, :);
    B_table(2*m, :) = -B_SE(m, :);
    
end
disp(B_table)

sheet_name = '파라메터(B)';
writematrix('B','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writematrix(B_table,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:D7') 


disp('Cbar')

Cbar_hat = reshape(meanc(Cbarm), M, K);
Cbar_SE = reshape(stdc(Cbarm), M, K);
Cbar_table = zeros(K, K);
for m = 1:M
    
    Cbar_table(2*m-1, :) = Cbar_hat(m, :);
    Cbar_table(2*m, :) = - Cbar_SE(m, :);
    
end
disp(Cbar_table)

sheet_name = '파라메터(Cbar)';
writematrix('Cbar','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writematrix(Cbar_table,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:G7') 

L_hat = reshape(meanc(Lm), M, M);
L_SE = stdc(Lm);
L_SE = reshape(L_SE, M, M);

L_table = zeros(K, M);
for m = 1:M
    
    L_table(2*m-1, :) = L_hat(m, :);
    L_table(2*m, :) = - L_SE(m, :);
    
end

disp(L_table)

sheet_name = '파라메터(L)';
writematrix('L','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writematrix(L_table,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:D7') 

%%
P = 1;
Wm_raw = Spec.Wm_raw;
USR_trend = MA_symetric(Wm_raw(:, 3), 24);
Wm = zeros(rows(Ym), cols(Wm_raw));
Wm(:, 1) = MA_symetric(Wm_raw(5:end, 1)-Wm_raw(4:end-1, 1), 4);
Wm(:, 2) = MA_symetric(Wm_raw(5:end, 2), 1)/1000000;
%Wm(:, 2) = Wm_raw(5:end, 2)/1000000;
% [Wm(:, 3), ~] = MA_symetric(Wm_raw(5:end, 3), 16);
% Wm(:, 3) = MA_symetric(Wm_raw(5:end, 3), 1);
%[~, Wm(:, 3)] = MA_symetric(Wm_raw(5:end, 3)-Wm_raw(4:end-1, 3), 2);
Wm(:, 3) = USR_trend(5:end, 1) - USR_trend(4:end-1, 1);
Wm(:, 4) = (Wm_raw(5:end, 4)-Wm_raw(4:end-1, 4))/1000;
  [Wm(:, 5)] = MA(Wm_raw(5:end, 5)-Wm_raw(4:end-1, 5), 2);
 %[Wm(:, 5), ~] = MA_symetric(log(Wm_raw(5:end, 5)), 8);
%Wm(:, 5) = (Wm_raw(5:end, 5))-(Wm_raw(4:end-1, 5));

%Wm(:, 4) = MA_symetric(Wm_raw(5:end, 4), 0)/1000;
Wm = demeanc(Wm(2:end, :));

index = [1, 5, 2, 3];
Prod_hat = Trend_hat(:, 1);
Nir_hat = Trend_hat(:, 3);

Y = Nir_hat(2:end) - Nir_hat(1:end-1);
Y = demeanc(Y(P+1:end, 1));

d_Prod = Prod_hat(2:end) - Prod_hat(1:end-1);
[d_Prod, ~] = MA_symetric(d_Prod, 5);
d_Money_Supply = Trend_hat(2:end, 2) - Trend_hat(1:end-1, 2);
[~, d_Money_Supply] = MA_symetric(d_Money_Supply, 1);
 X = [d_Prod(1:end-P, :), d_Money_Supply(1:end-P, :), Wm(1:end-P, index)];
%X = [Wm(1:end-P, index)];
X = demeanc(X);

% 전체
[bhat, sig2hat, stde, t_val, Yhat, ehat, varbhat, mY, TSS, RSS, R2] = OLSout(Y, X ,0);

beta_table_full = [bhat, t_val];
T = rows(Y);
betahatm = kron(ones(T, 1), bhat');
Y_hatm = X.*betahatm;
Y_hatm = [Y_hatm, ehat];


%%
sheet_name = '파라메터(beta)';
writematrix('전체기간','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writematrix('R squared','Results.xlsx','Sheet', sheet_name, 'Range', 'A8')
writematrix('estimate','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('t value','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix(beta_table_full,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:C10') 
writematrix(R2,'Results.xlsx','Sheet', sheet_name, 'Range', 'B8') 

sheet_name = 'NIR변동분해';
writematrix('$x_{t-1}^{g}$','Results.xlsx','Sheet', sheet_name, 'Range', 'B1') 
writematrix('$x_{t-1}^{m}$','Results.xlsx','Sheet', sheet_name, 'Range', 'C1') 
writematrix('$\Delta pop_{t-1}$','Results.xlsx','Sheet', sheet_name, 'Range', 'D1') 
writematrix('$\Delta sa_{t-1}$','Results.xlsx','Sheet', sheet_name, 'Range', 'E1') 
writematrix('$\Delta usr_{t-1}$','Results.xlsx','Sheet', sheet_name, 'Range', 'F1') 
writematrix('잔차항','Results.xlsx','Sheet', sheet_name, 'Range', 'G1') 
date = Spec.date;
writematrix('Date','Results.xlsx','Sheet', sheet_name, 'Range', 'A1') 
writecell(date(2:end),'Results.xlsx','Sheet', sheet_name, 'Range', 'A2:A1000') 
writematrix(Y_hatm,'Results.xlsx','Sheet', sheet_name, 'Range', 'B2:G1000') 

%% LW 결정요인 분석결과
Y = LW(2:end) - LW(1:end-1);
Y = demeanc(Y(P+1:end, 1));
[bhat, sig2hat, stde, t_val, Yhat, ehat, varbhat, mY, TSS, RSS, R2] = OLSout(Y, X ,0);

beta_table_LW = [bhat, t_val];

sheet_name = '파라메터(beta)';
writematrix(beta_table_LW,'Results.xlsx','Sheet', sheet_name, 'Range', 'B13:C20') 
writematrix(R2,'Results.xlsx','Sheet', sheet_name, 'Range', 'B19') 
