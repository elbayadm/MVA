for q=2
	if q==1 %question 1
		rbm=load('data/rbm_weights_8'); rbm=rbm.W_all;
		cd1=load('data/rbm-cd-1_weights_8'); cd1=cd1.W_all;
		cd10=load('data/rbm-cd-10_weights_8'); cd10=cd10.W_all;
		meths={'rbm','cd1','cd10'};

		n_input=8;
		n_hidden=8;
		nsamples=500;
	else
		cd1=load('data/rbm-cd-1_weights_10_20'); cd1=cd1.W_all;
		cd10=load('data/rbm-cd-10_weights_10_20.mat'); cd10=cd10.W_all;
		meths={'cd1','cd10'};

		n_input=20;
		n_hidden=20;
		nsamples=500;
	end
	% Gibbs sampling:
	for i=1:length(meths)
		meth=meths{i}; 
		bars=zeros(10*10,n_input);
		switch meth
			case 'rbm'
				W=rbm([1:n_input],n_input+[1:n_hidden]);
			case 'cd1'
				W=cd1([1:n_input],n_input+[1:n_hidden]);
			case 'cd10'
				W=cd10([1:n_input],n_input+[1:n_hidden]);
			end
		fprintf('\n%s &',meth);
		for K=10*(1:10)
			shifted=0;
			for ii=1:nsamples
				%random inital x:
				x = 2*double(rand(1,n_input)>.5)-1;
				for l=1:K
					%P(h=1|x):
					p_h=1./(1+exp(-2*x*W));
					h = 2*double(rand(size(p_h))<p_h)-1;
					p_x=1./(1+exp(2*h*W'));
					x = 2*double(rand(size(p_x))<p_x)-1;
				end
				shifted=shifted+iscorrect(x,n_input/2);
				if(ii<=10)
				bars(ii+10*(K/10-1),:)=x;
				end
			end
			fprintf('%.2f &',shifted/nsamples);
		end

		showbar(bars(:,:))
		set(gcf,'InvertHardcopy','off');
		text(repmat(-.25,1,10),.977-.103*(0:9),cellstr(num2str(10*(1:10)', 'K=%d')))
		print('-djpeg',sprintf('../images/bars_%s_%d_%d',meth,n_input,n_hidden))
	end
	if q==1
		%Plot P_ok:
		load('data/DL2_q1');
		figure,
		plot(rbm_OK,'k','linewidth',2)
		hold on
		plot(rbm_cd_1_OK,'r','linewidth',2);
		hold on,
		plot(rbm_cd_10_OK,'g','linewidth',2);
		legend({'rbm, brute force','rbm, CD1','rbm, CD10'},'location','southeast','fontsize',20);
		print('-djpeg','../images/rbm_oks')
	else
		load('data/DL2_q2');
		figure,
		plot(cd1_diffs,'k','linewidth',2)
		hold on
		plot(cd10_diffs,'r','linewidth',2);
		legend({'rbm, CD1','rbm, CD10'},'location','northeast','fontsize',20);
		print('-djpeg','../images/rbm_diffs')
	end
end
