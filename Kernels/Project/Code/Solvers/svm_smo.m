function [alpha, b, SV, varargout] = svm_smo ( K, Y, params)
% SVM smo solver    
	%  ------------------------ Params
    atol        = 1e-9;
    epoches		= 10;
    maxiter     = inf;
    tol         = params.tol;
	n 			= size(K,1);
    C			= params.C;
    coeff       = params.coeff;
    verb        = params.verb;
    id          = params.id;
	% ------------------------- Outputs
    alpha		= zeros(n, 1);
	b 			= 0;
	obj         = [0];    
    % Custom C -regularization for imbalanced set:
    Cpos = C; Cneg =C;
    C    = C*ones(n,1);
    if coeff
        Cpos    = coeff * Cneg;
        disp(Cpos)
        C(Y==1) = Cpos;
    end
    
    fprintf('----------------(SVM [%s])--------------- \n',id);
    fprintf('%12s: %d\n%12s: %d\n\n%12s: %.2f\n%12s: %.2f\n',...
              '#Samples',n,'#Positives',sum(Y==1),'C_neg',Cneg,'C_pos',Cpos);

    % ----------------------
    %       Main loop
    % ----------------------
    it 	   = 0;
    epoch  = 0;
    f      = zeros(n,1);
    tic,
    if params.verb
        fprintf('--------------------------------------- \n');
        fprintf('\t  iter  change   #SV   #bSV    obj \n');
        fprintf('--------------------------------------- \n');
    end
    while (epoch < epoches) && (it < maxiter)
        it  = it + 1 ;
    	change = 0;
        for i=1:n
    		Ei = f(i) + b - Y(i);
            if ((Y(i)*Ei < -tol)  && (alpha(i) < C(i))) || ((Y(i)*Ei > tol)  && (alpha(i) > C(i)))
    			% alpha(i) will be updated
    			% fetch another index j:
    			j = i;
                while j == i
    				j= randi([1 n],1);
                end
    			Ej = f(j) + b - Y(j);
    			% calculate L and H bounds for j so that alpha(j) is in [0 C(j)]
                if Y(i)==Y(j)
    				L = max(0, alpha(i)+alpha(j)-C(i));
    				H = min(C(j),alpha(i)+alpha(j));
                else
    				L = max(0,alpha(j)-alpha(i));
    				H = min(C(j), C(i)+alpha(j)-alpha(i));
                end
                if abs(L-H) < tol
                    continue
                end
				eta = 2*K(i,j)-K(i,i)-K(j,j);
                if eta >= 0
                    continue
                end
				% update alpha:
				aj = alpha(j) - Y(j)*(Ei-Ej)/eta;
				aj = min(max(aj,L),H);
                if abs(aj - alpha(j)) < tol
                    continue
                end
				ai = alpha(i) + Y(i)*Y(j)*(alpha(j)-aj);
				b1 = b - Ei - Y(i)*(ai - alpha(i))*K(i,i)...
				- Y(j)*(aj-alpha(j))*K(i,j);
				b2 = b -Ej - Y(i)*(ai - alpha(i))*K(i,j)...
				- Y(j)*(aj-alpha(j))*K(j,j);
                if (ai > 0) && (ai < C(i)) 
					b = b1;
                end
                if (aj > 0) && (aj < C(j))
					b = b2;
                end
                % Update the projections too:
                f = f + (ai - alpha(i))*Y(i)*K(i,:)'...
                    +  (aj - alpha(j))*Y(j)*K(j,:)';
                % Update the objective:
                obj(end+1) = obj(end) + (ai-alpha(i)) + (aj-alpha(j))...
                    -Y(i)*(ai-alpha(i))*sum(alpha.*Y.*K(:,i)) -Y(j)*(aj-alpha(j))*sum(alpha.*Y.*K(:,j));
				alpha(i) = ai ;
				alpha(j) = aj ;
				change = change + 1 ;
            end % Update alpha(i)
        end % Loop over samples
        SV =  alpha > atol;
        nSV= sum(SV);
        bSV =  sum(alpha == C);
        
        if ~mod(it,10) && verb
            fprintf('[%6s] %4i   %4i    %4i  %4i  %.2e\n', id, it, change, nSV, bSV, obj(end));
        end
        if ~change 
    		epoch = epoch + 1;
    	else
    		epoch = 0 ;
        end 
    end % Optim
    ts=toc;
    if epoch == epoches
        fprintf(2,'Solver converged in %.2f s\n',ts);
    else
        fprintf(2,'Solver did not converge [runtime : %.2f]\n',ts);
    end
    perf         = [];
    perf.nSV     = nSV;
    perf.bSV     = bSV;
    perf.iter    = it;
    perf.ctime   = ts;
    perf.cvg     = (epoch == epoches);
    perf.Cpos    = Cpos;
    perf.Cneg    = Cneg;
    perf.ns      = n;
    perf.pos     = sum(Y==1);
    perf.id      = id;
    perf.obj     = obj;
    varargout{1} = perf;
end