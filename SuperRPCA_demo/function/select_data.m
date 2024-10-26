function [data, map, lambda_a, lambda_b, threshold, N, superpixel_size, normalize] = select_data(number)
% Default parameter setting
lambda_a = 1; lambda_b = 0.1; threshold = 0.8; normalize = 0.3;
switch number
    case 1
        load('San_Diego.mat'); 
        N = 2; superpixel_size = 12; lambda_b = 1;
    otherwise
        error('No other data.');
end