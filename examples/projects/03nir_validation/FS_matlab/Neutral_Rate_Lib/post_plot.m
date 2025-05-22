function post_plot(MHm, maxac)

npara = cols(MHm);
[~, Acfm] = ineff(MHm, maxac);
racf = min(rows(Acfm), 100);
Acfm = Acfm(1:racf, :);
Acfm = [ones(1, cols(Acfm)); Acfm];
xa = 0:racf;
rs = 6;
nw = floor((npara - 1)/rs);

%% 파라메터의 수가 10보다 크면

if nw > 0
    for indnw = 1:nw
        scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/1.5 scrsz(3)/3 scrsz(4)/1.5]);
movegui(f);
        for indpara = rs*(indnw-1)+1:rs*indnw
            y = MHm(:, indpara);
            
            subplot(rs, 3, 3*(indpara -rs*(indnw-1) - 1) + 1);
            plot(y);
            xlim([0 rows(y)])
            ylabel(['para ', num2str(indpara)],'FontSize',11)
% ylabel(Y_labelm(indnw),'FontSize',11, 'interpreter', 'Latex')
            if indpara == 1
                title('trace plot')
            end
            
            subplot(rs, 3, 3*(indpara -rs*(indnw-1) - 1) + 2);
            histfit(y,40,'kernel')
            if maxc(y) > minc(y)
               xlim([minc(y) maxc(y)])
            end
            if indpara == 1
                title('empirical kernel density')
            end
            
            subplot(rs, 3, 3*(indpara -rs*(indnw-1) - 1) + 3);
            y = Acfm(:, indpara);
            bar(xa, y);
            xtick = 0:10:rows(y);
            set(gca,'XTick',xtick);
            xlim([0 racf+1])
            ylimlow = minc(y) - 0.05;
            ylim([ylimlow 1.05])
            if indpara == 1
                title('acutocorrelations')
            end
        end
    end
    
    scrsz = get(0,'ScreenSize');
f=figure('Position',[1 scrsz(4)/1.5 scrsz(3)/3 scrsz(4)/1.5]);
movegui(f);
    for indpara = rs*nw+1:npara
        y = MHm(:, indpara);
        
        subplot(rs, 3, 3*(indpara -rs*nw - 1) + 1);
        plot(y);
        xlim([0 rows(y)])
            ylabel(['para ', num2str(indpara)],'FontSize',11)
% ylabel(Y_labelm(indnw),'FontSize',11, 'interpreter', 'Latex')
        if indpara == rs*nw+1
            title('trace plot')
        end
        
        if maxc(y) > minc(y) 
           subplot(rs, 3, 3*(indpara -rs*nw - 1) + 2);
           histfit(y,40,'kernel')
           xlim([minc(y) maxc(y)])
           if indpara == rs*nw+1
               title('empirical kernel density')
           end
        end
        
        subplot(rs, 3, 3*(indpara -rs*nw - 1) + 3);
        y = Acfm(:, indpara);
        bar(xa, y);
        xtick = 0:10:rows(y);
        set(gca,'XTick',xtick);
        xlim([0 racf+1])
        ylimlow = minc(y) - 0.05;
        ylim([ylimlow 1.05])
        if indpara == rs*nw+1
            title('acutocorrelations')
        end
    end
    
end

%% 파라메터수가 10보다 작거나 같으면
if nw == 0
figure
    for indpara = 1:npara
        y = MHm(:, indpara);
        subplot(npara, 3, 3*(indpara - 1) + 1);
        plot(y);
        xlim([0 rows(y)])
        ylabel(['para ', num2str(indpara)],'FontSize',11)
        if indpara == 1
            title('trace plot')
        end
        
        subplot(npara, 3, 3*(indpara - 1) + 2);
        histfit(y,40,'kernel')
        xlim([minc(y) maxc(y)])
        if indpara == 1
            title('empirical kernel density')
        end
        
        subplot(npara, 3, 3*(indpara - 1) + 3);
        y = Acfm(:, indpara);
        bar(xa, y);
        xtick = 0:10:rows(y);
        set(gca,'XTick',xtick);
        xlim([0 racf+1])
        ylimlow = minc(y) - 0.05;
        ylim([ylimlow 1.05])
        if indpara == 1
            title('acutocorrelations')
        end
    end
end
end
