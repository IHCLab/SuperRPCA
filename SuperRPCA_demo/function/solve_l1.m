function [X]=solve_l1(b,lambda)
% soft threshold
%
% argmin 1/2||X-b||_{2}^2+\lambda||X||_{1}
%
X=sign(b).*max(abs(b) - lambda,0);