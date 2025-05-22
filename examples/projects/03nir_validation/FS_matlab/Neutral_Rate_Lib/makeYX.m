%% 종속, 설명변수 만들기 %%%%%%%%%%%%%%%%%%%%%%%%%%
function [Y0,YLm] = makeYX(Y,p)

k = cols(Y); % 변수의 수
T = rows(Y); % 시계열의 크기

Y0 = Y(p+1:T,:); % 종속변수

%설명변수(=Y의 과거값) 만들기
YL = zeros(T-p,p*k);
for i = 1:p
    YL(:,k*(i-1)+1:k*i) = Y(p+1 - i:T-i,:);
end

ki = p*k; % 각 식에 있는 설명변수의 수
kki = k*ki;

YLm = zeros(k, kki, T-p); % 설명변수를 3차원으로 새롭게 저장할 방
for t = 1:(T-p)
    xt = kron(eye(k), YL(t,:));
    YLm(:,:,t) = xt; % p by k
end

end

