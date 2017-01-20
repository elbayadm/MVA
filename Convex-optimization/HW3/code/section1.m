colors=jet(8);
rng(0,'twister');
doprint=false;

lambda=10;
eps=1e-10;

n=100;
d=10;

%Random sample
X=rand(n,d);
y=rand(n,1);
%% Section 1: 
% The barrier method for QP:
%Define QP parameters:
H=pinv(K'*K);
Q=H/2; p=-H*K'*y; A=[eye(N);-eye(N)]; b=lambda*ones(2*d,1);
% initialize with as strict. feasible point:
v0=zeros(d,1);

%Matlab's:
% x = quadprog(2*Q,p,A,b);

%% Plot the results for variant mu - outer iterations
set(0,'DefaultAxesFontSize',8)
figure('units','normalized','position',[.1 .1 .8 .8]),
nplots=3;
mu=[2,15,50,100];
for i=1:length(mu)
% run Barrier
[v_seq, loss, dGap, newton]=barr_method(Q,p,A,b,v0,eps,mu(i));
subplot(length(mu),nplots,1+nplots*(i-1))
plot(loss,'-*b')
xlabel('outer iteration');
ylabel('obj');
xlim([1 length(loss)])
title(sprintf('Objective function values\n(\\lambda=%.2f, \\mu=%.2f)',lambda,mu(i)))

subplot(length(mu),nplots,2+nplots*(i-1))
semilogx(dGap,'-*r')
grid on,
xlabel('outer iteration');
ylabel('duality gap');
xlim([1 length(loss)])
title(sprintf('Precision criterion\n(\\lambda=%.2f, \\mu=%.2f)',lambda,mu(i)))

subplot(length(mu),nplots,3+nplots*(i-1))
semilogx(loss-loss(end),'-*r')
grid on,
xlabel('outer iteration');
ylabel('gap');
xlim([1 length(loss)])
title(sprintf('Loss gap\n(\\lambda=%.2f, \\mu=%.2f)',lambda,mu(i)))
end
if doprint
print('-dpdf','images/Barrier_QP.pdf', '-opengl')
end
%% Plot with Newton iterations:
figure('units','normalized','position',[.1 .1 .8 .8]),
nplots=3;
mu=[2,15,50,100];
for i=1:length(mu)
% run Barrier
tic,
[v_seq, loss, dGap, newton]=barr_method(Q,p,A,b,v0,eps,mu(i));
timeit=toc;
fprintf('mu=%.2f - time= %.2f\n',mu(i),timeit);
arr1 = cumsum(newton);
arr2 = zeros(1,arr1(end));
arr2(arr1 - newton + 1) = 1;
newton_loss=loss(cumsum(arr2));

subplot(length(mu),nplots,1+nplots*(i-1))
plot(newton_loss,'-b')
xlabel('newton iteration');
ylabel('obj');
xlim([1 length(newton_loss)])
title(sprintf('Objective function values\n(\\lambda=%.2f, \\mu=%.2f)',lambda,mu(i)))

subplot(length(mu),nplots,2+nplots*(i-1))
newton_dGap=dGap(cumsum(arr2));
semilogx(newton_dGap,'-r')
grid on,
xlabel('newton iteration');
ylabel('duality gap');
xlim([1 length(newton_dGap)])
title(sprintf('Precision criterion\n(\\lambda=%.2f, \\mu=%.2f)',lambda,mu(i)))

subplot(length(mu),nplots,3+nplots*(i-1))
semilogx(newton_loss-newton_loss(end),'-r')
grid on,
xlabel('newton iteration');
ylabel('gap');
xlim([1 length(newton_loss)])
title(sprintf('Loss gap\n(\\lambda=%.2f, \\mu=%.2f)',lambda,mu(i)))
end

if doprint
print('-dpdf','images/Barrier_QP_nt.pdf', '-opengl')
end
