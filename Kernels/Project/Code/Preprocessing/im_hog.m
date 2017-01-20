function H = im_hog(im, CellSize, BlockSize,NumBins)
    % Unsigned - switch to 360 for signed gradient
    total_angles = 180.0;
    binwidth = total_angles/NumBins;
    overlap = ceil(BlockSize/2);
   
    % L2-hys clipping threshold
    th   = .2; 
    % For numerical issues
    tiny = 1e-8;
    
    % Mapping indices to bin centers
    bins_map = (binwidth/2:binwidth:total_angles)';

    % Compute the gradient in polar coordinates
    [Gdir, Gmag] = im_gradient(im,1);
    % Map to [0 180]
    Gdir(Gdir<0) = Gdir(Gdir<0) + 180;
    % Split the gradient in cells
    [cell_coords, nx ,ny] = split_windows(size(Gdir,1),size(Gdir,2),CellSize,CellSize);
    % 3D array to store the histograms
    histograms = zeros(ny,nx,NumBins);

    % --------------------------------------------------- Compute histograms
    for index=1:size(cell_coords,2)    
        h = zeros(1,NumBins);
        
        % cell coords
        xtart = cell_coords(1,index);
        xend = cell_coords(2,index);
        ystart = cell_coords(3,index);
        yend = cell_coords(4,index);

        % retrieve angles and magnitudes for all the pixels in the cell 
        angs = Gdir(ystart:yend,xtart:xend);
        mags = Gmag(ystart:yend,xtart:xend);
        % Bound the angles:
        low = floor(angs/binwidth + .5);
        up = low + 1;
        % Wrap over boundaries.
        low(low== 0) = 1;
        %up(up==(NumBins+1)) = NumBins;
        
        % retrieve the lower bin center value.
        lower_bins = bins_map(low);
        angs(angs < lower_bins) = binwidth - angs(angs < lower_bins);
        

        % Cast the votes
        upper_contribs = (angs-lower_bins)/binwidth;
        if find(upper_contribs > 1)
            warning('Something wrong')
            disp(angs)
            disp(low)
            disp(lower_bins)
        end
        lower_contribs = 1 - upper_contribs;
        upper_contribs = mags .*upper_contribs;
        lower_contribs = mags .*lower_contribs;
        
        % computing contributions for the current histogram bin by bin.
        for bin=1:NumBins
            % pixels that contribute to the bin with their lower part
            LPx = (low == bin);
            h(bin) = h(bin) + sum(lower_contribs(LPx));
            
            % pixels that contribute to the bin with their upper part
            UPx = (up == bin);
            h(bin) = h(bin) + sum(upper_contribs(UPx));
        end

        % Append to the 3D matrix
        row_offset    = floor(index/ny + 1);
        column_offset = mod(index-1,nx)+1;
        histograms(row_offset,column_offset,:) = h(1,:);
    end

    % --------------------------------------------------- block normalization
    hist_size = BlockSize*BlockSize*NumBins;
    descriptor_size = hist_size*(ny-BlockSize+overlap)*(nx-BlockSize+overlap);
    H = zeros(descriptor_size, 1);
    col = 1;
    row = 1;

    while row <= ny-BlockSize+1
        while col <= nx-BlockSize+1 
            % Retrieve the block histograms
            blockHists = histograms(row:row+BlockSize-1, col:col+BlockSize-1, :);
            % -------- L2-Hys normalization
            Nrm = norm(blockHists(:),2);
            normalized_blockHists = blockHists / (Nrm + tiny);
            normalized_blockHists (normalized_blockHists > th) = th;
            Nrm = norm(normalized_blockHists(:),2);
            normalized_blockHists = normalized_blockHists /(Nrm + tiny);
            offset = (row-1)*(nx-BlockSize+1)+col;
            st = (offset-1)*hist_size+1;
            en = offset*hist_size;

            H(st:en,1) = normalized_blockHists(:);
            col = col + overlap;
        end
        row = row + overlap;
        col = 1;
    end
end


function [windows , nx, ny]= split_windows(rows , cols ,w, h)
    % Split the image in windows of size [h,w]
    %  Output format:  [XS]  X start indices
    %                  [XE]  X end indices
    %                  [YS]  Y start indices 
    %                  [YE]  X end indices

    nx = floor(cols/w);
    ny = floor(rows/h);


    xstart(1:nx) = w * ((1:nx)-1) + 1;     
    xend(1:nx)   = w * min((1:nx),cols);   
    ystart(1:ny) = h * ((1:ny)-1) + 1;    
    yend(1:ny)   = h * min((1:ny),rows);  

    [YS,XS] = meshgrid(ystart,xstart);
    [YE,XE] = meshgrid(yend,xend);
    windows = [XS(:),XE(:),YS(:),YE(:)]';
end