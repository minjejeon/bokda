function disp_inef(inef)

n = rows(inef);
[x,~] = minresid(n,5);

disp('----------------------------------');
   for i = 1:x+1
       if i*5 <= n
           disp(inef(5*(i-1)+1:i*5)');
       else
           disp(inef(5*(i-1)+1:end)');
       end
   end
   disp('----------------------------------');
end
