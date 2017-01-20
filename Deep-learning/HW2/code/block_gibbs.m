function [x, p_h] = block_gibbs(x, W, L)
	for l=1:L
		%P(h=1|x):
		p_h=1./(1+exp(-2*x*W));
		h = 2*double(rand(size(p_h))<p_h)-1;
		p_x=1./(1+exp(2*h*W'));
		x = 2*double(rand(size(p_x))<p_x)-1;
	end
	p_h=1./(1+exp(-2*x*W));
