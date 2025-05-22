function [nmhblck,upp,low] = randupp(nmh,tp)

%/* sampling changepoint given transition probility, tp */
 if nmh > 1;
   upp_s = [zeros(nmh-1,1);1];
   for itr = 1:nmh;
      u = rand(1,1);
      if u > tp;   %/* if u>tp, then 'itr' becomes 'upp' */
         upp_s(itr) = 1;
      end
    end
  elseif nmh == 1
   upp_s = 1;
 end

%/* making the new 'upp' based on changepoint*/
 nmhblck = sumc(upp_s); %/* number of blocks */
 upp = zeros(nmhblck,1);
 itr = 1; jj=1;
 while itr <= nmh;
    if upp_s(itr) == 1;
       upp(jj) = itr;
       jj = jj + 1; 
    end
    itr = itr+1;
 end
 
  if  nmhblck > 1 % /* in case of multiple block */
   nb = upp-[0;upp(1:nmhblck-1)];
  else  %/* in case of single block  */
   nb = nmh;
  end

  upp = cumsum(nb);
 
%  /* making a new 'low' */
 low = [0;upp(1:nmhblck-1)];
 low = low + 1;
 
end