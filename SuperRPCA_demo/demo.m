clear all;  clc;
close all

addpath(genpath('dataset'));
addpath(genpath('function'));

% Data selection
data_number = 1;
[data, GT_map, lambda_a, lambda_b, threshold, N, S, m] = select_data(data_number);

% Data preprocessing
[L1, L2, M] = size(data); L = L1 * L2;
data = (data - min(data(:))) / (max(data(:)) - min(data(:)));

%% SuperRPCA algorithm (i.e., Algorithm 3)
    %% Collaborative superpixel representation detector (CSRD)
    [B_CSRD, time1]  = CSRD_optimization(data, S, m);
    %% RPCA with CSRD-based regularizer
    [result, time2] = SuperRPCA_optimization(data, B_CSRD, lambda_a, lambda_b, N, threshold);
    time = time1 + time2;

% Evaluate SuperRPCA performance
[PD_SuperRPCA, PF_SuperRPCA] = roc(GT_map(:), result(:));
AUC_SuperRPCA = - sum((PF_SuperRPCA(1: end - 1) - PF_SuperRPCA(2: end)) .* (PD_SuperRPCA(2: end) + PD_SuperRPCA(1: end - 1)) / 2);

% Show SuperRPCA detection result
figure; temp_show = data(:, :, [30 13 8]);
subplot(1, 3, 1); imshow(temp_show); title('False-color Image');
subplot(1, 3, 2); imshow(GT_map); title('Ground-truth Map');
temp_show = ImGray2Pseudocolor(result, 'hot', 255);
subplot(1, 3, 3); imshow(temp_show); title('SuperRPCA');
str = sprintf('AUC= %.4f\nTime= %.4f', AUC_SuperRPCA,time);
xlabel(str);


