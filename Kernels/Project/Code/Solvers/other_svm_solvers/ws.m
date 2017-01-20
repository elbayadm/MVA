function [B, gap] = ws(alpha, Y, gY, C, sz, tol)
% Select a working selection with the most violating pairs (Keerti, Gilbert 2002)
	delta = C - alpha;
    
	Ip = find((Y > 0) & (delta > tol));	%I_up +
	Im = find((Y < 0) & (alpha > tol));	%I_up _
	I = union(Ip, Im);
	
    Jp = find((Y > 0) & (alpha > tol));	%I_low +    
    Jm = find((Y < 0) & (delta > tol));	%I_low -
    J = union(Jp, Jm);

	% Violating pairs
	B_1 = gY(I);
	[~, order_I] = sort(B_1, 'descend');
	I = I(order_I);
	B_2 = gY(J);
	[~, order_J] = sort(B_2);
	J = J(order_J);

	% Select the pairs
	n = min(length(I), length(J));
	n = min(n, sz);
	B = union(I(1:n), J(1:n));
    
	% Optimality criterion
	gap = gY(I(1)) - gY(J(1));
	gap = abs(gap);
