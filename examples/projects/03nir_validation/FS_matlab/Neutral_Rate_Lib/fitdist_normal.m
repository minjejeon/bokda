function [x, y] = fitdist_normal(data)
   
pd = fitdist(data,'normal');

q = icdf(pd,[0.0013499 0.99865]); % three-sigma range for normal distribution
x = linspace(q(1),q(2));
if ~pd.Support.iscontinuous
    % For discrete distribution use only integers
    x = round(x);
    x(diff(x)==0) = [];
end

y = pdf(pd,x);

end