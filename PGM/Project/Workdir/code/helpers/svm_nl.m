
function  [beta,b,obj] = svm_nl(Y,lambda)
% -----------------------------------------------------
% Train a linear SVM using nonlinear conjugate gradient 
% -----------------------------------------------------
% a global variable K (the n x n kernel matrix) should be defined.  
% Y is the target vector (+1 or -1, length n). 
% LAMBDA is the regularization parameter ( = 1/C)
% © Olivier Chapelle, olivier.chapelle@tuebingen.mpg.de
% --Extract
opt=[];
opt.iter_max_Newton = 200;
opt.cg_it = 20;
opt.prec = 1e-6; 

	global K;
	n = length(K);
	beta = zeros(n+1,1); % The last component of beta is b.
	iter = 0;
	Kb = zeros(n,1); % Kb will always contains K times the current beta
	go = [Y; sum(Y)]; % go = -gradient at beta=0
	s = go; % Initial search direction
	Kgo = [K*Y; sum(Y)]; % We use the preconditioner [[K 0]; [0 1]]
	Ks = Kgo(1:end-1);   % Ks will always contain K*s(1:end-1)

	while 1
		iter = iter + 1;
		if iter > opt.cg_it * n
			warning(sprintf(['Maximum number of CG iterations reached. ' ...
			           'Try larger lambda']));
			break;
		end;

		% Do an exact line search
		[t,Kb] = line_search_nonlinear(s,Kb,beta(end),Y,lambda,0,Ks);
		beta = beta + t*s;

		% Compute new gradient and objective.
		% Note that the gradient is already "divided" by the preconditioner
		[obj, grad] = obj_fun_nonlinear(beta,Y,lambda,Kb); gn = -grad;
		%fprintf('Iter = %d, Obj = %f, Norm grad = %f     \n',iter,obj,norm(gn)); %Turn on for followup

		% Stop when the relative decrease in the objective function is small 
		if t*s'*Kgo < opt.prec*obj, break; end;

		Kgn = [K*gn(1:end-1); gn(end)];  % Multiply by the preconditioner 
		                         % -> Kgn is the real gradient

		% Flecher-Reeves update. Change 0 in 1 for Polack-Ribiere                             
		be = (Kgn'*gn - 0*Kgn'*go) / (Kgo'*go);
		s = be*s+gn;
		Ks = be*Ks + Kgn(1:end-1);

		go = gn;
		Kgo = Kgn;
	end;

	% The last component of the solution is the bias b.
	b = beta(end);
	beta = beta(1:end-1);


function [t, Kb] = line_search_nonlinear(step,Kb,b,Y,lambda,fullstep,Ks)
	% Given the current solution (as given by Kb), do a line sesrch in 
	% direction step. First try to take a full step if fullstep = 1.
	global K;
	training = find(Y~=0);
	act = find(step(1:end-1));  % The set of points for which beta change
	if nargin<7
	Ks = K(training,training(act))*step(act);
	end; 
	Kss = step(act)'*Ks(act); % Precompute some dot products
	Kbs = step(act)'*Kb(act);
	t = 0;
	Y = Y(training);
	% Compute the objective function for t=1
	out = 1-Y.*(Kb+b+Ks+step(end)); sv = out>0;
	obj1 = (lambda*(2*Kbs+Kss)+sum(out(sv).^2))/2;
	while 1
	out = 1-Y.*(Kb+b+t*(Ks+step(end)));
	sv = out>0;
	% The objective function and the first derivative (along the line)
	obj = (lambda*(2*t*Kbs+t^2*Kss)+sum(out(sv).^2))/2;
	g = lambda * (Kbs+t*Kss) - (Ks(sv)'+step(end))*(Y(sv).*out(sv)); 
	if fullstep & (t==0) & (obj-obj1 > -0.2*g)
	% First check t=1: if it works, keep it -> sparser solution
	t = 1;
	break;
	end; 
	% The second derivative (along the line)
	h = lambda*Kss + norm(Ks(sv)+step(end))^2;
	% fprintf('%d %f %f %f\n',length(find(sv)),t,obj,g^2/h);
	% Take the 1D Newton step
	t = t - g/h;
	if g^2/h < 1e-10, break; end;

	end;
	Kb = Kb + t*Ks;

function [obj, grad] = obj_fun_nonlinear(beta,Y,lambda,Kb)
	global K;
	out = Kb+beta(end);
	sv = find(Y.*out < 1);
	% Objective function...
	obj = (lambda*beta(1:end-1)'*Kb + sum((1-Y(sv).*out(sv)).^2)) / 2;
	% ... and preconditioned gradient
	grad = [lambda*beta(1:end-1); sum(out(sv)-Y(sv))];
	grad(sv) = grad(sv) + (out(sv)-Y(sv));
	% To compute the real gradient, one would have to execute the following line
	% grad = [K*grad(1:end-1); grad(end)];
