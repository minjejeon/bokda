function compute_rstar_FS(end_date, lib_path)

rng(123)
%path = 'C:\Users\bok\Desktop\work\matlab\NIR\KangShin_2024\Neutral_Rate_Lib';
%path = 'C:\Users\rolli\OneDrive - 고려대학교\바탕 화면\work\matlab\NIR\KangShin_2024\Neutral_Rate_Lib';
path = 'C:\Users\BOK\Desktop\pyprojects\nir_validation\FS_matlab\Neutral_Rate_Lib';
addpath(lib_path)

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

Ym = [GDPg, CPIg, CreditRatio, BaseRate];

%% Set up
Spec.date = date;
Spec.Ym = Ym;
Spec.No_FinCycle = 1;

%% 데이터 필터링 based on end_date
%end_date = '2024-06-30';
endtime = datetime(end_date, 'InputFormat', 'yyyy-MM-dd');
data_dates = datetime(string(date), 'InputFormat', 'yyyy-MM-dd');
data_dates = dateshift(data_dates, 'end', 'quarter');

filter_idx = data_dates <= endtime;

Spec.date = Spec.date(filter_idx);
Spec.Ym = Spec.Ym(filter_idx, :);


%%
tic
NIR_BOK(Spec)
toc

%% 금리 plot
load Rstm.mat Rstm
load Eyestm.mat Eyestm

%plot(BaseRate, '--k', 'LineWidth', 2);
%hold on;
%plot(meanc(Rstm), '-.r', 'LineWidth', 2);
%plot(meanc(Eyestm), 'b', 'LineWidth', 2);
%hold off;
%ylim([-2, 6]);
%legend({'기준금리', '실질중립금리', '명목중립금리'});

%% 금리 저장
startDate = datetime(2004, 3, 31);
endDate = datetime(end_date, 'InputFormat', 'yyyy-MM-dd');
Qdate = startDate:calmonths(3):endDate;

real_rstar = meanc(Rstm);
nominal_rstar = meanc(Eyestm);

T = table(Qdate', real_rstar, nominal_rstar, 'VariableNames', {'Date', 'RealNIR', 'NominalNIR'});
writetable(T, 'Rstar_FS.csv')

T_struct = table2struct(T);
json_str = jsonencode(T_struct);

fid = fopen('rstar_FS.json', 'w', 'n', 'utf-8');
if fid == -1
    error('Cannot create JSON file');
end
fwrite(fid, json_str, 'char');
fclose(fid);

end