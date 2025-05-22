%% Data Loading
Datafile = 'Macro_US';
Datasheet = 'Sheet1';
date = readcell(Datafile, 'sheet', Datasheet, 'Range', 'A43:A222');
Vari_index0 = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'B2:Z2');
is_KOR0 = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'B3:Z3');
is_growth0 = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'B4:Z4');
Vari_Names_KOR0 = readcell(Datafile, 'sheet', Datasheet, 'Range', 'B7:Z7');
Vari_Names_ENG0 = readcell(Datafile, 'sheet', Datasheet, 'Range', 'B9:Z9');
Raw_Data0 = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'B43:Z222');

% 자료분석에 포함여부
is_included = readmatrix(Datafile, 'sheet', Datasheet, 'Range', 'B1:Z1');