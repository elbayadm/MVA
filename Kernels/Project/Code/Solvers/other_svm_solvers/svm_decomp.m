function [alpha, b , SV, gap] = svm_decomp(K,Y, opts)
	tol		= opts.tol;
	atol	= opts.atol;
	maxiter	= opts.maxiter;
    
	n 		= size(K,1);
    C		= 1/opts.lambda/n;
	alpha	= sparse(n, 1);
	
    % Support vectors: 
    nSV= 0;
    SV = (alpha > atol);
	% Gradient
    g  		= ones(n, 1);
	gY 		= g .* Y;
	aY 		= alpha .* Y;
    
    it 		= 0;

    % Get a first violating pair:
	[A, B, Iup, Idown]=UpDown(Y, aY, C);
   	[H,i] = max(gY(Iup));
    i= Iup(i);
	[L,j] = min(gY(Idown));
    j= Idown(j);
	gap = H-L;

	fprintf('        iter     #SV        gap    \n');
	fprintf('-------------------------------------------- \n');
    fprintf('[%s]   %4i     %4i    %1.4e \n', ...
        opts.id, it,  nSV(end), gap(end));
    while (gap(end) > tol) && (it < maxiter)
		lambda = min([B(i) - aY(i),...
					  aY(j)-A(j),...
					  (H-L)/(K(i,i)+K(j,j)-2*K(i,j)) ]);
		% Update alpha and the gradients:
		g = g - lambda*Y.*(K(i,:)-K(j,:))';
		alpha(i) = alpha(i) + lambda*Y(i);
		alpha(j) = alpha(j) - lambda*Y(j);
        gY 		= g .* Y;
        aY 		= alpha .* Y;
		it = it + 1;
		SV = (alpha > atol);
		nSV(end+1) = sum(full(SV));

    	fprintf('[%s]   %4i     %4i    %1.4e \n', ...
        opts.id, it,  nSV(end), gap(end));

    	%Get a new pair:
        [A, B, Iup, Idown]=UpDown(Y, aY, C);
        [H,i] = max(gY(Iup));
        i= Iup(i);
        [L,j] = min(gY(Idown));
        j= Idown(j);
		gap(end+1) = H-L;	    
    end
    b= (H+L)/2;
    if gap(end) < tol
        fprintf('The SVM WS converged')
    else
        fprintf('The SVM WS did not converge')
    end
   

function [A, B, Iup, Idown]=UpDown(Y, aY, C)
	B = C*(Y==1); %Upper bound
	A = -C*(Y==-1); %Lower bound
    % Iup
	Iup = find (aY < B);
	Idown = find ( aY > A);