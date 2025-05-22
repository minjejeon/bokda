function kde2d_contour(X, Y, X_label, Y_label, x_lim, y_lim)

scrsz = get(0,'ScreenSize');
figure('Position',[1 scrsz(4)/4 scrsz(3)/4 scrsz(4)/2])
scatter(X,Y);
hold on
data = [X, Y];
[~,density,X,Y]=kde2d(data);
contour(X,Y,density,40)
if nargin<=2
   ylabel('YData', 'FontSize', 17);
   xlabel('XData', 'FontSize', 17);
else
   ylabel(Y_label, 'FontSize', 17);
   xlabel(X_label, 'FontSize', 17);
   
end
set(gca,'fontsize',15)
if nargin> 4
   xlim(x_lim);
   ylim(y_lim);
end

end
