function Smoothed_Prob = Smoother(FilterP, p00, p11)

Smoothed_Prob = FilterP; % T by 2
T = rows(FilterP); % 표본크기
p01 = 1 - p00;
p10 = 1 - p11;
for t = (T-1):-1:1
    %% P(st = 0, st1 = 1)
    Prb01_T = Smoothed_Prob(t+1, 2)*p01*FilterP(t, 1);
    Prb01_T = Prb01_T/(FilterP(t, 1)*p01 + FilterP(t, 2)*p11);
    
    %% P(st = 0, st1 = 0)
    Prb00_T = Smoothed_Prob(t+1, 1)*p00*FilterP(t, 1);
    Prb00_T = Prb00_T/(FilterP(t, 1)*p00 + FilterP(t, 2)*p10);
    
    %% 저장하기
    Prb0_T = Prb01_T + Prb00_T;    
    Smoothed_Prob(t, 1) = Prb0_T;
    Smoothed_Prob(t, 2) = 1 - Prb0_T;
    
end

end