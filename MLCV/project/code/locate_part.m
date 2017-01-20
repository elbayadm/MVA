function box  = locate_part(map,th,sc)
	% Inputs:
	% map : messages to the part
	% th  : threshold level 
	% Outputs: 
	% each row is a candidate part [minx miny maxx maxy score]
	if nargin<2
		th = 0;
	end	
	box = [];
	CC = bwconncomp(map>th);
	if CC.NumObjects
		for k = 1:CC.NumObjects
			component  =  CC.PixelIdxList{k};
			[Y,X] = ind2sub(size(map), component);
			X= X/sc; Y = Y/sc;
			x1 = min(X); x2 = max(X) ; y1 = min(Y) ; y2 = max(Y);
			if (y2-y1)>5 & (x2-x1)>5 % large enough	
				box(end+1,:) = [x1 y1 x2 y2 max(map(component)) sc];
			end
		end
	else
		box = [];
	end
end