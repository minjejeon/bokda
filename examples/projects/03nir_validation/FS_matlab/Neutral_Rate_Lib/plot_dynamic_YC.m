function plot_dynamic_YC(YCm,tau, Datem, frame)

if nargin < 4
    frame = 1/20;
end
maxy = 1.1*maxc1(YCm);

xtick = tau;

ytick = 0:2:ceil(maxy);

ntau = rows(tau);

n = 10;
   
  
for i = 2:rows(YCm)
    
    yc1 = YCm(i-1, :)';
    yc2 = YCm(i, :)';
    
    yc = zeros(ntau, n);
    interval = (yc2 - yc1)/(n-1);
    for j = 1:ntau
        intj = interval(j);
        if abs(intj) > 0
            a = yc1(j):intj:yc2(j);
        else
            a = yc1(j);
        end
        yc(j,:) = a';
    end
    
    
    h = plot(tau, yc2, '-o', tau, yc,'--');
    set(h(1),'linewidth',4);
    set(h(2),'linewidth',1);
    set(h(1),'color', [0 0 0]);
    xlabel('Maturity (month)','FontSize',12)
    ylabel('(%)','FontSize',12)
    ylim([0 maxy])
    xlim([0 tau(end)])
    set(gca,'XTick',xtick,'FontSize',12);
    set(gca,'YTick',ytick);
%     datai = [num2str(Datem(i,1)),'³â ',num2str(Datem(i,2)),'¿ù'];
    datai = num2str(Datem(i,1));
    h_legend = legend(datai);
    set(h_legend,'location','Southeast','FontSize',12);

%     drawnow;
    pause(frame);
    
end

end
