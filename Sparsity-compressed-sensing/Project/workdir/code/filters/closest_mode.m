function output = closest_mode(I,width,sx,sv,T)
% Closest mode filter
% smoothed, locally-weighted histogram

[h , w] = size(I);
[s_list,Ds] = smoothed_histograms(I,width,sx,sv,T,'derivative');

% Loop over the pixels to interpolate the derivative and find the negative-going zero crossings
tic,
interp1 = @(x, x1, y1, x2, y2) (y2-y1)/(x2-x1)*(x-x1)+y1;
output  = zeros([h w]);
D       = zeros([h w]);
for  x = 1:w
    for y = 1:h
        % Pixel at (y,x)
        Dp = squeeze(Ds(y,x,:));
        % interpolate the derivatives
        [~,i] = find((s_list - I(y,x))>= 0,1,'first');
        assert(~isempty(i),'Error locating the pixel intensity')
        x1 = map(s_list,i-1);
        y1 = map(Dp,i-1);
        x2 = s_list(i);
        y2 = Dp(i);
        D(y,x)= interp1(I(y,x),x1,y1,x2,y2);  
        % find the modes:
        pos = Dp >= 0;
        neg = Dp < 0;
        neg = [neg(2:end); neg(1)];
        modes = find(neg.*pos);
        s_modes = s_list(modes)' + Dp(modes)./(Dp(modes) - map(Dp,modes+1))...
            .*(map(s_list,modes+1)-s_list(modes)');
        if numel(modes)
            if D(y,x)>=0
                % choose the first mode greater than Ip
                [~,i] = find(s_modes' >=I(y,x),1,'first');
            else 
                % choose the last mode smaller than Ip
                [~,i] = find(s_modes' <=I(y,x),1,'last');
            end
            if isempty(i)
                fprintf(2,'(%d,%d): assigned first mode- D=%.2f\n',y,x,D(y,x));
                disp(s_modes')
                i=1;
            end
            output(y,x) = s_modes(i);
        else
            warning('Pixel (%d,%d) left blank',y,x)
        end
        
    end
end
fprintf('Modes & derivatives computation : %.2fs\n',toc)
end
 