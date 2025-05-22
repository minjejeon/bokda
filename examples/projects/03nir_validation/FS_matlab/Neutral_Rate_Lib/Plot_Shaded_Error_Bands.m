function H = Plot_Shaded_Error_Bands(x, y)
% Written by 강규호. 아무나 임의로 수정해서 사용할 수 있습니다.
% y = N by 3, 또는 N by 5, 또는 N by 7
% x = N by 1,  X 축

% 예를 들어, y의 열의 수가 5이고, 3번째 열이 평균, 1과 5열이 95% 신뢰구간, 2열과 4열이 80% 신뢰구간으로
% 설정해서 그림을 그릴 수 있습니다.
% 1기-ahead, 2기-ahead기,.., H기-ahead 예측치와 신뢰구간을 동시에 그릴 때 유용할 수 있습니다.

[rs, cs] = size(y); % 열의 수
mc = ceil(cs/2);
m = y(:, mc); % 예측치
y = y';
x = x';

%% 평균선 그리기
figure
H.mainLine = plot(x, m', 'b-o', 'LineWidth', 4);
xlim([x(1)-0.5 x(end)+0.5]); % X 축의 범위
xtick = x;
set(gca,'XTick', xtick);
title([num2str(1), '기부터 ', num2str(rs), '기까지의 예측분포'])
xlabel('Forecast horizon','FontSize',11)
ylabel('(%)','FontSize',11)

%% 첫 번째 신뢰구간 그리기
hold on;
col = [0, 0, 0]; % 검은색
patchSaturation=0.6;
patchColor=col+(1-col)*(1-patchSaturation);
px=[x,fliplr(x)];
py=[y(mc-1,:), fliplr(y(mc+1,:))];
H1.patch = patch(px,py,1,'FaceColor',patchColor,'EdgeColor','none', 'facealpha',patchSaturation);
legend([H.mainLine, H1.patch], ...
    '예측치', '95%',  ...
    'Location', 'Northwest');

%% 두 번째 신뢰구간 그리기
if cs > 3
    patchSaturation=0.45;
    patchColor=col+(1-col)*(1-patchSaturation);
    hold on;
    py=[y(mc-2,:), fliplr(y(mc-1,:))];
    H2.patch = patch(px,py,1,'FaceColor',patchColor,'EdgeColor','none', 'facealpha',patchSaturation);
    py=[y(mc+1,:), fliplr(y(mc+2,:))];
    H2.patch = patch(px,py,1,'FaceColor',patchColor,'EdgeColor','none', 'facealpha',patchSaturation);
    legend([H.mainLine, H1.patch, H2.patch], ...
        '예측치', '80%','95%', ...
        'Location', 'Northwest');
    
    %% 세 번째 신뢰구간 그리기
    if cs > 5
        patchSaturation=0.3;
        patchColor=col+(1-col)*(1-patchSaturation);
        hold on;
        py=[y(mc-3,:), fliplr(y(mc-2,:))];
        H3.patch = patch(px,py,1,'FaceColor',patchColor,'EdgeColor','none', 'facealpha',patchSaturation);
        py=[y(mc+2,:), fliplr(y(mc+3,:))];
        H3.patch = patch(px,py,1,'FaceColor',patchColor,'EdgeColor','none', 'facealpha',patchSaturation);
        legend([H.mainLine, H1.patch, H2.patch, H3.patch], ...
            '예측치', '50%', '80%','95%', ...
            'Location', 'Northwest');
        
    end
end

end