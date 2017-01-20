function elt = map_matrix(M,I)
	% Given indexes I and input matrix M(3D array) return (M(i))i in the same shape as I
	% where I is the 2rd dimension index
	[X Y Z] = size(M);
	assert((size(I,1)==X) & (size(I,2)==Y));
	elt= zeros([X Y]);
	for x= 1:X
		for y=1:Y
			elt(x,y) = map(M(x,y,:),I(x,y));
		end
	end

