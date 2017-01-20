function alpha = subproblem(alpha, Y, Q, C, B, N, tol)
	H = Q(B, B);
	Q_BN = Q(B, N);
	f = (Q_BN * alpha(N) - 1)';

	% Set quadrprog parameters:
	beq = - alpha(N)' * Y(N);
	alpha_B = alpha(B);
	n = length(alpha_B);
	e = ones(n, 1);
	options = optimoptions('quadprog',...
	    'Algorithm','interior-point-convex','Display','off');

	% Solve and update alpha:
	alpha_B = quadprog(H, f, [], [], Y(B)', ...
	                   beq, 0*e, C*e, alpha_B, options);

	alpha(B) = alpha_B;
