function countm = CountTransitions(sm)

% Counting the # of Regime switches 
nreg = maxc(sm); % 레짐의 수
countm = zeros(nreg,nreg); % count를 저장할 방
n = rows(sm);

for t = 2:n;
    SL = sm(t-1);
    St = sm(t);
    countm(SL,St) = countm(SL,St) + 1;
end

end