function [alpha, b , SV, F, gap] = svm_ws(K,Y, opts)
	tol		= opts.tol;
	atol	= opts.atol;
	maxiter	= opts.maxiter;
	sz 		= opts.sz;
    
	n 		= size(K,1);
    C		= 1/opts.lambda/n;
	alpha	= sparse(n, 1);
	Q		= diag(Y) * K * diag(Y);
	
    % Objective funtion:
	F 		= sum(alpha) - 0.5 * alpha' * Q * alpha;
	% Gradient
    g  		= ones(n, 1);
	gY 		= g .* Y;
    % Support vectors
	SV 		= (alpha > atol);
	nSV     = 0;
	
    it 		= 0;

	% Initiate the working set:
	[B, gap] = ws(alpha, Y, gY, C, sz, atol);
	N = setdiff((1:n)', B);

	fprintf('        iter        F          #SV        gap    \n');
	fprintf('-------------------------------------------- \n');
    df = 1 ;
	while (gap(end) > tol) && (it < maxiter) && (df > 0)
		% Solve the supbroblem on the working selection:
		alpha = subproblem(alpha, Y, Q, C, B, N, tol);
		SV = (alpha > atol);
	    nSV(end+1) = full(sum(SV));
	    F(end+1) = sum(alpha) - 0.5 * alpha' * Q * alpha; 
        if length(gap) > 20 
            df = diff(gap);
            df = sum((df(end-19:end) > tol));
        end
	    g = 1 - Q * alpha;
	    gY = g .* Y;

	    % New working selection:
	    [B, gap(end+1)] = ws(alpha, Y, gY, C, sz, tol);
    	N = setdiff((1:n)', B);
    	fprintf('[%s]   %4i    %1.7e    %4i    %1.4e \n', ...
        opts.id, it, F(end), nSV(end), gap(end));
        b = max(gY(B))-min(gY(B));
	    it = it + 1;
    end
    if gap(end) < tol
        fprintf('The SVM WS converged')
    else
        fprintf('The SVM WS did not converge')
    end
   

