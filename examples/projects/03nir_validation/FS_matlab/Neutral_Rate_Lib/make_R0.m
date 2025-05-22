%Macro to compute unconditional variance of the factors
function [B0, R0] = make_R0(mu, G, omega)

 k = rows(G);
 B0 = (eye(k) - G)\mu;
 k2 = k^2;
 G2 = kron(G,G);
 eyeG2 = eye(k2)-G2;
 omegavec = reshape(omega,k2,1);
 R00 = (eyeG2)\omegavec; %  vec(R00)
 R0 = reshape(R00,k,k)';
 R0 = (R0 + R0')/2;

end