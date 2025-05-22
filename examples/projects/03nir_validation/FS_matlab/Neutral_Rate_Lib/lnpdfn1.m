% log pdf of standard normal */
function [lpdf] = lnpdfn1(e)

c = -0.5*log(2*pi);  	

lpdf = c - 0.5*e .* e;   

end
