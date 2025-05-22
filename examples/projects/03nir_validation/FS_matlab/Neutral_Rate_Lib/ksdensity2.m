function [xgrid,ygrid,pdf] = ksdensity2(dat,m);

% ksdensity2: bivariate kernel smoother using normal kernel
% dat is n-by-2:  Rows correspond to observations and columns correspond to variables 
% m is number of points in grid (square grid assumption)
% Note: if no output arguments specified, then output is a surface plot

n = size(dat,1);
x = dat(:,1); y = dat(:,2);
sx = std(x); sy = std(y);
h1 = sx * n ^(-1/6);
h2 = sy * n ^(-1/6);
h = [h1; h2]; h = h*ones(1,n);   % 2-by-n
pdf = zeros(m+1,m+1);

ax = min(x) - sx;        % the lower limit of the grid
bx = max(x) + sx;        % the upper limit of the grid
incx = (bx-ax)/m;

ay = min(y) - sy;        % the lower limit of the grid
by = max(y) + sy;        % the upper limit of the grid
incy = (by-ay)/m;

xgrid = ax:incx:bx;
ygrid = ay:incy:by;

for j = 1:1:m+1
    for i = 1:1:m+1
        x = [xgrid(1,j); ygrid(1,i)];   % current grid point 2-by-1
        x1 = x*ones(1,n);               % 2-by-n
        dev = ( x1-dat')./ h;
        p = sum(mvnpdf(dev'));
        pdf(i,j) = p;
    end;
end;

if nargout == 0;
    surf(xgrid,ygrid,pdf);
end;