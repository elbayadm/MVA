%% Section 3: Proximal methods:
rng(0,'twister');
colors=jet(8);
doprint=true;

lambda=10;
eps=1e-5;
n=100;
d=10;

X=rand(n,d);
y=rand(n,1);
%% standard:
[w_seq,loss_seq, iter]=prox_method(X,y,lambda,eps);
set(0,'DefaultAxesFontSize',14)
figure('units','normalized','position',[0 0 .7 .3],'PaperPositionMode','auto','PaperOrientation','landscape'),
semilogx(loss_seq-min(loss_seq),'-*r')
xlim([1 length(loss_seq)]);
xlabel('iteration')
ylabel('f(w_t)-f^*')
title('Proximal method - convergence')

if doprint
  print('-dpdf','images/proximal.pdf', '-opengl')
end

%% accelrated:
[w_seq,loss_seq, iter]=prox_acc_method(X,y,lambda,eps);

figure('units','normalized','position',[0 0 .7 .3],'PaperPositionMode','auto','PaperOrientation','landscape'),
semilogx(loss_seq-min(loss_seq),'-*r')
xlim([1 length(loss_seq)]);
xlabel('iteration')
ylabel('f(w_t)-f^*')
title('FISTA - convergence')

if doprint
  print('-dpdf','images/proximal_acc.pdf', '-opengl')
end
