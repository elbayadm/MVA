function output = morphology(I,width,sx,sv,T,r)
% Morphology filter
% r = 5%  Dilation : opening
% r = 95% Erosion  : closing
% r = 50% Median

% smoothed, locally-weighted histogram
[s_list, Rs] = smoothed_histograms(I,width,sx,sv,T,'integral');
step = s_list(2)-s_list(1);
tic,
% spot 1/2 between Rs(i) and Rs(i+1)
[Rs_prev,I] = max(Rs.*(Rs<r),[],3);
Rs_next = map_matrix(Rs,I+1);
s_prev = (I-1)*step;
output = s_prev + (r - Rs_prev)./(Rs_next - Rs_prev)*step;
fprintf('Median computation : %.2fs\n',toc)
