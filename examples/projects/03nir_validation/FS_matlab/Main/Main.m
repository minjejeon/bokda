clear;
clc;
rng(123)
path = 'C:\Users\bok\Desktop\work\matlab\NIR\KangShin_2024\Neutral_Rate_Lib';
addpath(path)

%% Data Loading
Datafile = 'nir_data_all';
Datasheet = 'raw_data';

date = readcell(Datafile, 'sheet', Datasheet, 'Range', 'A178:A260');
GDP = readmatrix(Datafile, 'Sheet', Datasheet, 'Range', 'B174:B260');
CPI = readmatrix(Datafile, 'Sheet', Datasheet, 'Range', 'C174:C260');
BaseRate = readmatrix(Datafile, 'Sheet', Datasheet, 'Range', 'F178:F260');
Credit = readmatrix(Datafile, 'Sheet', Datasheet, 'Range', 'H174:H260');
NGDP = readmatrix(Datafile, 'Sheet', Datasheet, 'Range', 'G174:G260');

GDPg = (GDP(5:end) - GDP(1:end-4)) ./ GDP(1:end-4) * 100;
CPIg = (CPI(5:end) - CPI(1:end-4)) ./ CPI(1:end-4) * 100;
CreditNgdp = 100*log(Credit ./ NGDP);
CreditRatio = CreditNgdp(5:end) - CreditNgdp(1:end-4);

%% Data Loading
Datafile = 'NIR_Data4_per_capita';   % 중립금리 추정에 사용한 데이타 엑셀파일
Datasheet = 'Raw_Data_KOR';      % Sheet 명
date = readcell(Datafile, 'sheet', Datasheet, 'Range', 'A18:A95');  % 날짜
GDPg = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'C18:C95'); % 거시자료
CPIg = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'E18:E95'); % 거시자료
BaseRate = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'F18:F95'); % 거시자료
CredRatio = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'L18:L95'); % 거시자료

Ym = [GDPg, CPIg, CredRatio, BaseRate];
Pop = readmatrix(Datafile, 'sheet', 'Pop_KOR', 'Range', 'L5:L25'); % 거시자료

Pop_KOR = Pop;
save Pop_KOR Pop_KOR

Pop = Trans_Y_Q(Pop);
Pop = Pop(1:rows(Ym));

%% Set up
Spec.date = date;
Spec.Ym = Ym;
Spec.Pop = Pop;
Spec.No_FinCycle = 1;

%%
tic
NIR_BOK(Spec)
toc

%% 금리 plot
load Rstm.mat Rstm
load Eyestm.mat Eyestm

plot(BaseRate, '--k', 'LineWidth', 2);
hold on;
plot(meanc(Rstm), '-.r', 'LineWidth', 2);
plot(meanc(Eyestm), 'b', 'LineWidth', 2);
hold off;
ylim([-2, 6]);
legend({'기준금리', '실질중립금리', '명목중립금리'});