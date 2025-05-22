% /* This function randomizes blocks  */
% /* Independent regime change is adapted */
% /* tp: transition probability, 1/(1-tp) is the implied-average number of parameters per block */
% nmhblck = number of blocks
function [indv,low,upp,nmhblck] = randblocks(indv,tp)

%/* randomizing permutation */
indv = rndper(indv);

% randomizing the number of blocks
[nmhblck,upp,low] = randupp(rows(indv),tp);

end