function [spSegs,numSuperpixels] = Superpixel(img,a,b)
%SUPERPIXEL Summary of this function goes here
%   Detailed explanation goes here
%% reorder and rescale data into 2-D array
[numRows,numCols,numSpectra] = size(img);
scfact = mean(reshape(sqrt(sum(img.^2,3)),numRows*numCols,1));
img = img./scfact;
imgVec = reshape(img,[numRows*numCols numSpectra]);

%% compute superpixels
spSegs = vl_slic(single(img),a,b); 

numSuperpixels = double(max(spSegs(:)))+1;
end

