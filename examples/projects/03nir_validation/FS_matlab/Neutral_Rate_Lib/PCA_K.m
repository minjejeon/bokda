function [PCm, eigen_valm, Vm, CORM ] = PCA_K(Y, is_COV)
% Y = T by k
% PCm = T by k
if nargin == 1
    is_COV = 0;
end

% 1 단계: 상관계수행렬 계산하기
if is_COV == 0
   CORM = corrcoef(Y);
elseif is_COV == 1
   CORM = cov(Y);
end

% 2 단계: 특성근, 특성벡터 계산하기
[V, D] = eig(CORM);

% V = 특성벡터, D의 대각값들 = 특성근
   
% 3 단계
eigen_val = diag(D); % 3 by 1
[eigen_val, index] = sort(eigen_val, 'descend'); % 특성근 정렬하기
Vm = V(:, index);% 특성벡터 정렬하기

eigen_valm = [eigen_val, eigen_val/sumc(eigen_val)];

% 4 단계: PC 계산하기
if is_COV == 0
   stand_Y = standdc(Y);
   PCm = stand_Y*Vm; % T by 3
elseif is_COV == 1
   PCm = Y*Vm; % T by 3
end

disp(['특성근 = ', num2str(eigen_val'/sumc(eigen_val))]);
end