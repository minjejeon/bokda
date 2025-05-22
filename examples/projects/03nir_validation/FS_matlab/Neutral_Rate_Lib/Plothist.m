function Plothist(X, x_label, x_title)

n = cols(X);
if nargin < 2
    x_label = 'X';
    x_title = 'empirical kernel density';
end

if nargin < 3
    x_title = 'empirical kernel density';
end

if n == 1
    figure
    histfit(X, 50, 'kernel')
    title(x_title)
    xlim([minc(X) maxc(X)])
    xlabel(x_label,'FontSize',11)
end

cs = min(5, n);
rs = ceil(n/5);
if n > 1
    figure
    
    for i = 1:n
        
        subplot(rs, cs, i)   
        y = X(:, i);
        histfit(y, 50, 'kernel')
        title(x_title(i));
        xlim([minc(y) maxc(y)])
        if nargin > 1
            xlabel(x_label(i));
        else
            xlabel(['var ', num2str(i)], 'FontSize', 11)
        end      
    end
end

end