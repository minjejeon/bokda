% Proposal in the Hamiltonian MH method based on Gradient
function [x00,F_val1,F_val0,g,Hinv,H] = Hamiltonian(FUN,CONSTR,argnew,index,Sn)

    g = -Gradpnew1(FUN,argnew,index,Sn);
	H = -FHESSnew1(FUN,argnew,index,Sn);
    H = real(H);

	if rows(H) > 1;
	
  		Hinv = invpd(H);

    	if isempty(Hinv) == 1;
    	  	[Hc,indef,err] = cholmod(H);
    	  	H = Hc'*Hc;
    	  	Hinvc = inv(Hc);
    	  	Hinv = Hinvc*Hinvc';
      end
    	
  else
		H = maxc(abs(H)|1e-016);
		Hinv = 1/H;
  end

  	db = -Hinv*g'; 		% the change in estimate
   	s = 1;
   	x00 = argnew;
    F_val0 = FUN(x00,Sn);
    
   	x00(index) = x00(index) + s*db/2;
    
    F_val1 = FUN(x00,Sn);
    
    while CONSTR(x00) == 0 && isnan(F_val1) == 0

    	    x00 = argnew;
         	x00(index) = x00(index) + s*db/2;
            F_val1 = FUN(x00,Sn);
             	if s < 1e-6 
          	        x00 = argnew;
         		    disp('close to constraint (minimum step reached)');
                    s = 0;
               end
             s = s/2;
    end
    
    H = -FHESSnew1(FUN,x00,index,Sn);
    H = real(H);
    
    if rows(H) > 1;
	
  		Hinv = invpd(H);

    	if isempty(Hinv) == 1;
    	  	[Hc,indef,err] = cholmod(H);
    	  	H = Hc'*Hc;
    	  	Hinvc = inv(Hc);
    	  	Hinv = Hinvc*Hinvc';
      end
    	
  else
		H = maxc(abs(H)|1e-016);
		Hinv = 1/H;
    end
  
end