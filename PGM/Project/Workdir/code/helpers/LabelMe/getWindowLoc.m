function [loc_index,obj_coords,location,valid] = getWindowLoc(objects,names,image_size,heights)
    % loc_index category index in objects list
    % obj_coords: coords as in Hoeim (2004) (Ly,log(Lz))
    % location [cx cy sx sy] : center coords and window size
    % valid Fishing for errors!
    N = length(objects);
    valid = false(N,1);
    loc_index = zeros(N,1);
    location = zeros(N,4);
    for o=1:length(objects)
        obj_label = objects(o).name;
        [tf, var_index] = ismember(obj_label, names);
        if (tf)
            valid(o) = true;
            loc_index(o) = var_index;
            [X,Y] = getLMpolygon(objects(o).polygon);
            % Center coords:
            cx = (min(X)+max(X))/2; cy = (min(Y)+max(Y))/2;
            % Window dimension
            sx = max(X)-min(X); sy = max(Y)-min(Y);
            location(o,:) = [cx cy sx sy];
        end
    end

    loc_index(~valid) = [];
    location(~valid,:) = [];
    % Drop sx= window's width
    obj_coords = image2WorldCoords(location(:,[1,2,4]),image_size,heights(loc_index));

    % Drop the unused Lx:
    obj_coords = obj_coords(:,2:3);


function xyz = image2WorldCoords(uvr,image_size,R)

    % Transform from image to world coordinates.

    f = 1;
    u0 = image_size(1)/2;
    v0 = image_size(2)/2;
    num_coords = size(uvr,1);
    uvr = double(uvr) - repmat([u0, v0, 0],num_coords,1);
    uvr = uvr/v0;
    xyz = [uvr(:,1), uvr(:,2), f*ones(num_coords,1)];
    %R the real world object height
    relative_size = R./uvr(:,3);
    xyz = xyz.*repmat(relative_size, 1, 3);

    % Take the log of z since log(z) is Gaussian
    xyz(:,3) = log(xyz(:,3));

