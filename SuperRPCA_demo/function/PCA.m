function [Xd,C]=PCA(X,N)
[M L ] = size(X);
d = mean(X,2);
U = X-d*ones(1,L);
[eV D] = eig(U*U');
C = eV(:,M-N+1:end);
Xd = C'*(X-d*ones(1,L)); 
