%=====================================================================
% Programmer: Jhao-Ting Lin
% E-mail: q38091534@gs.ncku.edu.tw
% Date: 2024/10/22
%======================================================================
%  Input:
%  Data is L1-by-L2-by-M hyperspectral data cube.
%  B_CRD is the reconstructed background.
%  lambda_a, lambda_b, N, and threshold are the parameters.
%----------------------------------------------------------------------
%  Output:
%  Result is L1-by-L2 detection result matrix whose entries are normalized.
%  Time is the computation time (in sec.).
%========================================================================

function [result, time] = SuperRPCA_optimization(data, B_CRD, lambda_a, lambda_b, N, threshold)

[L1, L2, M] = size(data);
L = L1 * L2;
data = reshape(data, L, M)';

% Initialization
Y = data;
S = zeros(N, L);
A = zeros(M, L);
J1 = zeros(N, L);
J2 = zeros(M, L);
J3 = zeros(M, L);
J4 = zeros(N, L);

mu = 1 / norm(Y);
mu_bar = mu * 1e7;
maxIter = 100;
rho = 1.1;

timer=tic;

[E] = compute_basis(B_CRD, N);
S_CRD = E' * B_CRD;

%% Algorithm 2
iter = 0;
converged = false;
while ~converged
    iter = iter + 1;

    % Update U1 U2 (non-separable)
    EI = [E, eye(M)];
    SsubD1_stack_AsubD2 = [S - J1; A - J2];
    U12 = ((EI' * EI) + mu * eye(M + N)) \ (EI' * Y + mu * SsubD1_stack_AsubD2);
    U1 = U12(1: N,:);
    U2 = U12(N + 1: end, :);

    % Update U3
    temp = A - J3;
    U3 = solve_l1(temp(:), lambda_a / mu);
    U3 = reshape(U3, M, L);

    % Update U4
    U4 = (lambda_b * S_CRD + mu * (S - J4)) / (lambda_b + mu);

    % Update S
    S_old = S;
    S = (U1 + J1 + U4 + J4) / 2;

    % Update A
    A_old = A;
    A =  (U2 + J2 + U3 + J3) / 2;

    % Update J1~5
    J1 = J1 - S + U1 ;
    J2 = J2 - A + U2;
    J3 = J3 - A + U3;
    J4 = J4 - S + U4;

    % Update mu
    mu = rho * mu;
    mu = min(mu * rho, mu_bar);

    % Stop criterion
    pri_res = sqrt((norm(S - U1,'fro')) ^ 2+ (norm(A - U2, 'fro')) ^ 2+ (norm(A - U3, 'fro')) ^ 2 + (norm(S - U4, 'fro')) ^ 2); % primal residual
    pri_tol = 0.001 * max(sqrt((norm([U1, U4], 'fro')) ^ 2 + (norm([U2, U3], 'fro')) ^ 2), sqrt((norm([S, S], 'fro')) ^ 2 + (norm([A, A], 'fro')) ^2));
    dual_res = mu * sqrt((norm(repmat(S - S_old, 1, 2), 'fro')) ^ 2 + (norm(repmat(A - A_old, 1, 2), 'fro')) ^ 2); % dual residual
    dual_tol = 0.001 * mu * sqrt(norm([J1, J4], 'fro') ^ 2 + norm([J2, J3], 'fro') ^ 2);
    if ((pri_res <= pri_tol) && (dual_res <= dual_tol)), break; end

    if ~converged && iter >= maxIter
        converged = 1 ;
    end
end

for j = 1:L
    result(j) =  sqrt(sum(A(:, j) .^ 2));
end
result = reshape(result', L1, L2, 1);
result = (result - min(result(:))) / (max(result(:)) - min(result(:)));

% Capped-square (CS) transform
result = CS_transform(result, threshold);

time = toc(timer);
