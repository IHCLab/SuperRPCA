function [E] = compute_basis(X, N)
[M, ~] = size(X);
U = X;
[eV, lambda] = eig(U * U');
diag(lambda);
E = eV(:, M-N+1: end);