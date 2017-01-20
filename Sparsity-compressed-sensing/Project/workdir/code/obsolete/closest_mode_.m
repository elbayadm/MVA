function [output, stats] = closest_mode(I,width,sx,sv)
% Closest mode filter
% smoothed, locally-weighted histogram
tiny = 1e-6;
tol  = 1e-3;

stats = [];
[h , w] = size(I);
[fs, Ds, s_list, W] = smoothed_histograms(I,width,sx,sv);
step = s_list(2) - s_list(1);
m    = numel(s_list); 
%% Loop over the pixels to interpolate the derivative and find the negative-going zero crossings
tic,
% spot I between s(i) and s(i+1)
SS = bsxfun(@times, ones([h,w,m]), reshape(s_list,[1,1,m]));
% s1(hxw) s2(hxw)....sm(hxw)

SS = ((SS - repmat(I,[1 1 m])) <= 0);
SS = bsxfun(@times, SS, reshape(s_list,[1,1,m]));
[S_previous,SS] = max(SS,[],3);
S_next  = S_previous + step;
D_previous = map_matrix(Ds,SS);
D_next = map_matrix(Ds,SS+1);

D = (D_next - D_previous)./(S_next - S_previous).*(I-S_previous);
fprintf('Derivative computation : %.2fs\n',toc)

tic,
% spot the negative-going zero crossings:
J_next = diff(sign(Ds),1,3);
J_next = cat(3,zeros([h w]),J_next<0);
II = J_next.*bsxfun(@times, ones([h w m]), reshape(1:m,[1,1,m]));
output = zeros([h w]);
for  x = 1:w
    for y = 1:h
        %indices of the shifts following the modes
        modes = nonzeros(II(y,x,:));
        s_next = s_list(modes)';
        s_prev = map(s_list,modes-1);
        d_next     = squeeze(Ds(y,x,modes));
        d_prev     = map(Ds(y,x,:),modes-1);
        s          = s_prev + d_prev ./(d_prev - d_next).*(s_next- s_prev);
        if D(y,x)>=0
            % choose the first mode greater than Ip
            [~,i] = find(s' >=I(y,x),1,'first');
            %assert(~isempty(i),'all modes are smaller than I!')
        else 
            % choose the last mode smaller than Ip
            [~,i] = find(s' <=I(y,x),1,'last');
            %assert(~isempty(i),'all modes are larger than I!')
        end
        if isempty(i)
            i =1;
        end
        output(y,x) = s(i);
    end
end
fprintf('Modes computation : %.2fs\n',toc)

stats.DI = D;
stats.Ds = Ds;
stats.fs = fs;
stats.ss = s_list;
stats.W  = W;
end
