function [alpha] = solve_weight(V, indice, neighbors, N_i)
V_j = zeros(N_i - 1, N_i);

superpixel_data = V(3: end, :);
DR_superpixel_data = PCA(superpixel_data, N_i - 1);
DR_S = DR_superpixel_data(:, neighbors);
DR_c = DR_superpixel_data(:,indice);

for i = 1: N_i
    g(:, i) = compute_bi(DR_S, i, N_i);
    h(i, 1) = max(g(:, i)' * DR_S);
end

if N_i == 2
    alpha = (h - g' * DR_c) ./ (h - (g .* DR_S)' );
else
    alpha = (h - g' * DR_c) ./ (h - sum(g .* DR_S)');
end

if all(alpha(:) >= 0)
    return;
else
    x = mean(DR_S, 2); %point on S ; better initialization
    d_0=zeros(N_i - 1, 1);
    D_j=zeros(N_i - 1, N_i);
    mu = 1 / norm(DR_S);
    maxIter = 100;

    %% Algorithm 1
    iter = 0;
    converged = false;
    while ~ converged
        iter = iter + 1;
        x_old = x;
        % Update v
        v_0 = (DR_c + mu * (x - d_0)) / (mu + 1);
        for j = 1: N_i
            n_j = x - D_j(:, j);
            if  g(:, j)' * n_j <= h(j)
                v_j = n_j;
            else
                v_j = n_j-(((g(:, j)' * n_j - h(j)) / (norm(g(:, j)') ^2)) .* g(:, j));
            end
            V_j(:, j) = v_j;
        end
        % Update x
        x = (v_0 + d_0 + sum(V_j + D_j, 2)) / (N_i + 1);
        % Update d
        d_0 = d_0 - x + v_0;
        D_j = D_j - repmat(x, 1, N_i) + V_j;

        % Check primal and dual residual
        pri_res = sqrt((norm(x - v_0,'fro')) ^2 + (norm(repmat(x, 1, N_i)- V_j,'fro')) ^2); % primal residual
        pri_tol = 0.001 * max(sqrt((norm(v_0, 'fro')) ^2 + (norm(V_j, 'fro')) ^2), sqrt((norm(x, 'fro')) ^2 + (norm(repmat(x, 1, N_i),'fro')) ^2 ));
        dual_res = mu * sqrt((norm(repmat(x - x_old, 1, N_i + 1), 'fro')) ^2); % dual residual
        dual_tol = 0.001 * mu * norm([d_0, D_j], 'fro');
        if ((pri_res <= pri_tol) && (dual_res <= dual_tol)), break; end

        % Check iter num
        if ~converged && iter >= maxIter
            converged = 1 ;
        end

    end

    % Compute the abundance
    if N_i == 2
        alpha = (h - g' * x) ./ (h - (g .* DR_S)');
    else
        alpha = (h - g' * x) ./ (h - sum( g .* DR_S)');
    end
    sum(alpha);
end

% Subprogram 1
function [bi] = compute_bi(a0, i, N)
Hindx = setdiff([1: N], [i]);
A_Hindx = a0(:, Hindx);
A_tilde_i = A_Hindx(:, 1: N - 2) - A_Hindx(:, N - 1) * ones(1, N - 2);
bi = A_Hindx(:, N - 1) - a0(:, i);
bi = (eye(N - 1) - A_tilde_i * (pinv(A_tilde_i' * A_tilde_i)) * A_tilde_i') * bi;
bi = bi / norm(bi);
return;
