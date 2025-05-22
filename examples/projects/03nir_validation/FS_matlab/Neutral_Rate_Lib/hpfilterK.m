function [Trend, Cycle] = hpfilterK(Ym, m)

Trend = hpfilter(Ym, 'Smoothing', m);
Cycle = Ym - Trend;

end