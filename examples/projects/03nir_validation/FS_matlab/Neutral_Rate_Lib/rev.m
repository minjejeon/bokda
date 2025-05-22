% REV 
% REV(x) reverses the elements of x
function y = rev(x)

y = x;
if rows(x) > 1
  i = rows(x):-1:1;
  y = x(i,:);   
end

end