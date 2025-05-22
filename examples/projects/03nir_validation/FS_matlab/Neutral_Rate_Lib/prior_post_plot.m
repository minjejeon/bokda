function prior_post_plot(priorm, postm, x_label)

y1 = priorm;
y2 = postm;
y = [y1, y2];
xlimit = [minc1(y) maxc1(y)]; % (x축) 분포의 범위

%% 밀도추정
[f1, x1] = ksdensity(y1);

[f2, x2] = ksdensity(y2);

hPlot1 = plot(x1,f1);
set(hPlot1,'Color','k','LineStyle','--','Linewidth',2)
hold on;
hPlot2 = plot(x2,f2);
set(hPlot2,'Color','b','LineStyle','-','Linewidth',2)

xlabel(x_label,'interpreter','Latex','FontSize',12)
ylabel('density','FontSize',12)
xlim(xlimit)
legend('prior','posterior','location','NorthEast','fontsize',12);
set(gca,'FontSize',12);

end