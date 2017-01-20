function [alpha, d, b, SV, varargout] = SMO_MKL ( K, Y, params)
% SVM Solver with multiple kernels
    % K is a cell array of kernels.
	%  ------------------------ Params
    atol        = 1e-9;
    epoches		= 10;
    maxiter     = inf;
    lambda      = .1;
    tol         = params.tol;
    nKernels    = length(K);
	nSamples 	= size(K{1},1);
    C			= params.C;
    coeff       = params.coeff;
    verb        = params.verb;
    id          = params.id;
	% ------------------------- Outputs
    alpha		= zeros(nSamples, 1);
    %[alpha{:}]  = deal(zeros(nSamples, 1));

	b 			= 0;
	   
    % Custom C -regularization for imbalanced set:
    Cpos = C; Cneg =C;
    C    = C*ones(nSamples,1);
    if coeff
        Cpos    = coeff * Cneg;
        C(Y==1) = Cpos;
    end

    % Kernels coeffs:
    d = ones(nKernels,1)/2;
    delta = zeros(nSamples,nKernels);
    fprintf('----------------(SVM MLK [%s])--------------- \n',id);
    fprintf('%12s: %d\n%12s: %d\n\n%12s: %.2f\n%12s: %.2f\n',...
              '#Samples',nSamples,'#Positives',sum(Y==1),'C_neg',Cneg,'C_pos',Cpos);

    % ----------------------
    %       Main loop
    % ----------------------
    it 	   = 0;
    epoch  = 0;
    f      = zeros(nSamples,1);
    tic,
    if params.verb
        fprintf('------------------------------------ \n');
        fprintf('\t  iter  change   #SV   #bSV  \n');
        fprintf('------------------------------------ \n');
    end
    while (epoch < epoches) && (it < maxiter)
        it  = it + 1 ;
    	change = 0;
        for i=1:nSamples
    		Ei = f(i) + b - Y(i);
            if ((Y(i)*Ei < -tol)  && (alpha(i) < C(i))) || ((Y(i)*Ei > tol)  && (alpha(i) > C(i)))
    			% alpha(i) will be updated
    			% fetch another index j:
    			j = i;
                while j == i
    				j= randi([1 nSamples],1);
                end
    			% calculate L and U bounds for j so that alpha(j) is in [0 C(j)]
                s = - Y(i)*Y(j);
                if s == 1
    				L = max(-alpha(i), -alpha(j));
    				U = min(C(j)-alpha(j),C(i)-alpha(i));
                else
    				L = max(-alpha(i),alpha(j)-C(j));
    				U = min(C(i)-alpha(i),alpha(j));
                end
                %disp('Bounds:')
                %disp([L U])
                if abs(U-L) < tol
                    continue
                end
				
                % Computing the optimal step:
                aa = zeros(nKernels,1);
                bb = aa; 
                for k =1:nKernels
                    aa(k)= K{k}(i,i)+K{k}(j,j)-2*K{k}(i,j);
                    bb(k)=Y(i)*(K{k}(i,:)-K{k}(j,:))*(alpha.*Y);
                end
                AA = sum(aa.^2);
                BB = sum(aa.*bb);
                CC = sum((bb.^2)+ lambda* aa.*d);
                DD = sum(bb.*d);
                % Solve the cubic equation:
                R = roots([AA, 3*BB , 2*CC , 2*lambda*(DD-(s+1))]);
                eta = 0;
                for rt = R'
                    if isreal(rt) 
                        %fprintf('rt = %.2f',rt);
                        %fprintf(' [L U] = [%.2f %.2f]\n',L,U);
                        if (rt >= L) && (rt <= U) && (abs(rt) > abs(eta))
                            eta = rt;
                        end
                    end
                end
                
                if abs(eta) < tol
                    continue
                end
                %fprintf('Chosed eta = %.4e\n',eta);
				% update alpha:
                alpha(i) = alpha(i) + eta;
                alpha(j) = alpha(j) + s*eta;

                % Update the kernesl coefficients:
				d = d + 1/2/lambda*eta*bb;
                
                % Update the projections too:
                % 
                diff = zeros(nSamples,1);
                for k=1:nKernels
                    diff = diff + d(k)*(K{k}(:,i)-K{k}(:,j));
                end

                f = f + 1/2/lambda*eta*delta*bb + eta*Y(i)*diff;

                % Update the bias term:
                b1 = Y(i) - f(i);
                b2 = Y(j) - f(j);
                if (alpha(i)>0) && (alpha(i)<C(i))
                    b = b1;
                elseif (alpha(j)>0) && (alpha(j)<C(j))
                    b = b2;
                else
                    b = (b1+b2)/2;
                end
                
				change = change + 1 ;

                % Update delta
                for k =1:nKernels
                    delta(:,k)= delta(:,k) + eta*Y(i)*(K{k}(:,i)-K{k}(:,j));
                end

            end % Update alpha(i)
        end % Loop over samples
        SV =  alpha > atol;
        nSV= sum(SV);
        bSV =  sum(alpha == C);
        if ~mod(it,10) && verb
            fprintf('[%6s] %4i   %4i    %4i  %4i\n', id, it, change, nSV, bSV);
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
    perf.ns      = nSamples;
    perf.pos     = sum(Y==1);
    perf.id      = id;
    varargout{1} = perf;
end