function [fx,h,Xrange,Yrange]=KDE2_eff(Xi, Yi, X_label, Y_label)

% Revised by Taesam Lee
% for large data set
%
% Kernel Density Estimate for 2 vairable
% for diagonal type h and multivariate normal kernel
% Each "Smoothing parameter" h is estimated from the SJ method 
% unless it is provided
% The X and Y range is devided 30 along the min and max 
% of historical data as a default
%
% Used subfunction : bandwidth_SJ.m
%
% Input
% Xi Yi : Historical Data (n*1) vector with the same data length
% h : bandwidth
% Xrange, Yrange : nr*1 vector with same length
%
% Output
% fx : Estiamted 2 variable density matrix (nr*nr) or 30*30(default)
% h  : Sheather and Jones 
% Xrange, Yrange - min~max with 30 (default)
%
% Programmed by Taesam Lee (02.16.2009)
% Reference : Simonoff JS(1996)-Smoothing methods in Statistics


% nL=length(Xi);

    rax=range(Xi);
    ray=range(Yi);
    Xrange=linspace(min(Xi)-rax*0.1,max(Xi)+rax*0.1,100);
    Yrange=linspace(min(Yi)-ray*0.1,max(Yi)+ray*0.1,100);


    [a,b,h(1)]=ksdensity(Xi,0);
    [a,b,h(2)]=ksdensity(Yi,0);

nX=length(Xrange);
nY=length(Yrange);
for ix=1:nX
    for iy=1:nY
        u1=(Xrange(ix)-Xi)/h(1);
        u2=(Yrange(iy)-Yi)/h(2);
        u=[u1,u2]';
        Kd_u=1/(2*pi)^(2/2)*exp(-1/2*sum(u.^2));
        %for is=1:nL
        %    Kd_u(is)=1/(2*pi)^(2/2)*exp(-1/2*u(:,is)'*u(:,is));
        %end
        fx(ix,iy)=mean(Kd_u)/prod(h);
    end
end
figure
scatter(Xi,Yi);
hold on,
contour(Xrange,Yrange,fx',20);
if nargin<=2
   ylabel('YData');
   xlabel('XData');
else
   ylabel(Y_label);
   xlabel(X_label);
end
% title('Biv. KDE (contour) with scatterplot of X&Y');
end

