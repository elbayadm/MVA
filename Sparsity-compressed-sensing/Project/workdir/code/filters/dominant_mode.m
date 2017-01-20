function output = dominant_mode(I,width,sx,sv,T)
% Dominant mode filter with smoothed, locally-weighted histogram
[h , w] = size(I);
[s_list,fs, Ds, Rs] = smoothed_histograms(I,width,sx,sv,T,{'hist','derivative','integral'});
m = length(s_list);

% Loop over the pixels to interpolate the derivative and find the modes and antimodes.
tic,
interp1 = @(x, x1, y1, x2, y2) (y2-y1)/(x2-x1)*(x-x1)+y1;
output  = zeros([h w]);
D       = zeros([h w]);
for  x = 1:w
    for y = 1:h
        % Pixel at (y,x)
        Dp = squeeze(Ds(y,x,:));
        Rp = squeeze(Rs(y,x,:));
        % interpolate the derivatives
        [~,i] = find((s_list - I(y,x))>= 0,1,'first');
        assert(~isempty(i),'Error locating the pixel intensity')
        x1 = map(s_list,i-1);
        y1 = map(Dp,i-1);
        x2 = s_list(i);
        y2 = Dp(i);
        D(y,x)= interp1(I(y,x),x1,y1,x2,y2);  
        % modes and antimodes:
        pos = Dp >= 0;
        neg = Dp < 0;
        % negtaive going zero crossings
        modes = find([neg(2:end); neg(1)].*pos);
        s_modes = s_list(modes)' + Dp(modes)./(Dp(modes) - map(Dp,modes+1))...
            .*(map(s_list,modes+1)-s_list(modes)');
        if numel(modes)>1
            % positive going zero crossings
            antimodes = find([pos(2:end); pos(1)].*neg); 
            antimodes = [1 antimodes'];
            weights   = diff(Rp(antimodes));
            %fprintf('(%dx%d): |modes| = %d, |antimodes|=%d\n',y,x,numel(modes),numel(antimodes));
            [~ , i] = max(weights);
        else
            i =1;
        end
        output(y,x) = s_modes(i);  
        %% visualization:
%         if ((y == 149) & (x == 257))
%             disp('Visualization...')
%             figure(2);clf;
%             set(2,'units','normalized','position',[.1 .1 .4 .3])
%             fp = squeeze(fs(y,x,:));
%             plot_hist(I(y,x),output(y,x),s_list,s_modes,fp, Dp,antimodes)
%             drawnow,
%         end
    end
end
fprintf('Modes & derivatives computation : %.2fs\n',toc)
end
 
