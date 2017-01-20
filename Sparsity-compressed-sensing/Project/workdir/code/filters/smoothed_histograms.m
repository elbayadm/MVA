function [s_list,varargout] = smoothed_histograms(I,width,sx,sv,samp,method)
% Compute smoothed histogram and its derivative
% Inputs:
% I      : input image
% width  : width of the weights matrix
% sx     : width of spatial kernel
% sv     : width of values kernel
tic,
% Gaussian kernel:
[gridX, gridY] = meshgrid( 0 : (2*width), 0 : (2*width));
gridX = gridX - width;
gridY = gridY - width;
gridRSquared = ( gridX .* gridX + gridY .* gridY);
GaussianKernel = @(s) exp(gridRSquared/2/s^2);

% Spatial kernel
W = GaussianKernel(sx); W = W/sum(W(:));

% shifts:
m = floor(samp/sv);
s_list = linspace(0,1,m);
step = s_list(2)-s_list(1);
if ~iscell(method)
    method = {method};
end

II = bsxfun(@minus, repmat(I,[1 1 m]), reshape(s_list,[1,1,m]));

if sum(ismember({'hist','derivative'},method))
    K = exp( -II.^2 / (2*sv^2) );
    size(K)
    K(1,4,5)
    K= K./repmat(sum(K,3),[1 1 m]);
end


varargout = {};
for i = 1:length(method)
    switch method{i}
        case 'hist'  
            varargout{end+1} = convn(K,W,'same');

        case 'derivative'
            dK = II/sv^2.*K;
            varargout{end+1} = convn(dK,W,'same');

        case 'integral'
            C = sv*sqrt(pi/2) * (1 + erf(-II/sv/sqrt(2)));
            % normalize:
            Rs = convn(C,W,'same');
            Z = Rs(:,:,end);
            varargout{end+1} = Rs./repmat(Z,[1 1 length(s_list)]);
    end
end

fprintf('Histogram computation : %.4fs\n',toc)
