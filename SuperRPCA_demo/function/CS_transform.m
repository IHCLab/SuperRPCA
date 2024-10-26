function [result_transform] = CS_transform(result, threshold)

index_lower = find(result(:) < threshold); 
index_higher = find(result(:) >= threshold);

nonlinear_transform_lower = ((1 / threshold) * result(index_lower)) .^ 2;
result_transform = result(:);

result_transform(index_lower) = nonlinear_transform_lower; 
result_transform(index_higher) = 1;

result_transform = reshape(result_transform, size(result, 1),size(result, 2));