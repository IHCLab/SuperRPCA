%=====================================================================
% Programmer: Jhao-Ting Lin
% E-mail: q38091534@gs.ncku.edu.tw
% Date: 2024/10/22
%======================================================================
%  Input:
%  Data is L1-by-L2-by-M hyperspectral data cube.
%  Superpixel_size (S) and normalize (m) are two parameters related to superpixel segmentation method.
%----------------------------------------------------------------------
%  Output:
%  B_CSRD is M-by-L reconstructed background.
%  Time is the computation time (in sec.).
%========================================================================

function [B_CSRD, time] = CSRD_optimization(data, superpixel_size, normalize)
tic;
[L1, L2, M] = size(data);
pixel = L1 * L2;
data = reshape(data, pixel, M)';

% SLIC
datad = PCA(data, 3);
datad_ori = reshape(datad', L1, L2, []);
[spSegs, K]  = Superpixel(datad_ori, superpixel_size, normalize); 
spSegs = spSegs + 1;

% Remove null superpixel
valid_sp = unique(spSegs);
null_sp_index = setdiff(1: K, valid_sp);
diff_map = zeros(size(spSegs, 1), size(spSegs, 2));
for i = null_sp_index
    diff_map(find(spSegs > i)) = diff_map(find(spSegs > i)) - 1;
end
spSegs = double(spSegs) + diff_map;
K = K - length(null_sp_index);


% Obtain spatial–spectral superpixel centroids (SCs)
for i = 1:K
    coordinate{i, 1}(:, 1) = find(spSegs == i);  %(:,1) pixel number
    coordinate{i, 1}(:, 2) = mod(coordinate{i, 1}(:, 1), L1); %(:,2) pixel y coordinate
    if (coordinate{i, 1}(:, 2) == 0)
        coordinate{i, 1}(:, 2) = L1;
    end
    coordinate{i, 1}(:, 3) = ceil(coordinate{i, 1}(:, 1) ./ L2); %(:,3) pixel x coordinate
    L(:, i) = round(mean(coordinate{i, 1}(:, 2: 3))); % coordinate average
    C(:, i)= mean(data(:, coordinate{i, 1}(:, 1)), 2)'; % spectral average
end
V=[L' C']';
spSegs_padding = padding(spSegs, L1, L2);

for i=1:size(V,2)
    % Neighborhood collection mechanism
    neighbors = [spSegs_padding(L1 + min(coordinate{i}(:, 2)) - 1, L2 + V(2, i)), ...
            spSegs_padding(L1 + max(coordinate{i}(:, 2)) + 1, L2 + V(2, i)), ...
            spSegs_padding(L1 + V(1, i), L2 + min(coordinate{i}(:, 3)) - 1), ...
            spSegs_padding(L1 + V(1, i), L2 + max(coordinate{i}(:, 3)) + 1)]; % Combine up, down, left, and right neighbors
    
    neighbors = unique(neighbors(neighbors ~= i)); % Remove self and duplicates
    N_i = length(neighbors);
    P_i = V(3: end, neighbors);

    % Solve weight
    if N_i == 1
        w_i = 1;
    else
        [w_i] = solve_weight(V, i, neighbors, N_i);
    end
    
    c_i(:, i) = P_i * w_i;  

end

% Reconstruct the background
for i=1: size(V, 2)
    B_CSRD(:, coordinate{i, 1}(:, 1)) = repmat(c_i(:, i), 1, length(coordinate{i, 1}(:, 1)));
end

time=toc;

% Subprogram 1
function [spSegsTest] = padding(spSegs, a, b)
spSegsTest = zeros(3 * a, 3 * b);
spSegsTest(a + 1: 2 * a, b + 1: 2 * b) = spSegs;
spSegsTest(a + 1: 2 * a, 1: b) = spSegs(:, b: -1: 1);
spSegsTest(a + 1: 2 * a, 2 * b + 1: 3 * b) = spSegs(:, b: -1: 1);
spSegsTest(1 : a, :) = spSegsTest(2 * a: -1: (a + 1), :);
spSegsTest(2 * a + 1: 3 * a, :) = spSegsTest(2 * a: -1: (a + 1), :);
return;


