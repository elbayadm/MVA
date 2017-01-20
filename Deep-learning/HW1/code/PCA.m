do_print=1;
doviz=0;
%%
[Dtrain,Dtest]  = load_digit7;
[nsamples,ndimensions] = size(Dtrain);
ntest=size(Dtest,1);
meanDigit = mean(Dtrain,1);
%Show the main image
meanImage = reshape(meanDigit,[28,28]);
if doviz 
figure,imshow(meanImage);
end

%Performing PCA on Dtrain:
x = Dtrain - repmat(meanDigit,nsamples,1);
S = (x'*x)/(nsamples-1);
xtest = Dtest - repmat(meanDigit,ntest,1);

%PCA error assessment:
E=zeros(1,10);
Etrain=zeros(1,10);

for num_retain = 1:10 % number of eigenvectors to retain
[Evec_retained,Evalm] = eigs(S,num_retain);
Eval = diag(Evalm); % find the e-vals and e-vecs

%Train error
gap=x-x*(Evec_retained*Evec_retained');
Etrain(num_retain)=sum(arrayfun(@(idx) norm(gap(idx,:)), 1:size(gap,1)));

gap=xtest-xtest*(Evec_retained*Evec_retained');
E(num_retain)=sum(arrayfun(@(idx) norm(gap(idx,:)), 1:size(gap,1)));

end

if do_print
figure(1)
plot(log(E),'-bx','linewidth',2);
hold on,
plot(log(Etrain),'-rx','linewidth',2);
xlabel('|Chosen Eigenvectors|','fontsize',20);
ylabel('log(Error)','fontsize',20);
legend('Test error','Train error');
print(1,'-dpdf','images/tocrop/PCA_test.pdf', '-opengl');
end

%Showing the eigen vectors:
if doviz
figure,
nr=3;
nc=floor(num_retain/nr)+1*(mod(num_retain,nr)~=0);
for k=1:num_retain
eig_vec=reshape(Evec_retained(:,k)+meanDigit',[28,28]);
subplot(nr,nc,k);
imshow(eig_vec);
end

x_lowerdim = x*Evec_retained; % lower dimensional representation
x_reconstruction = x_lowerdim*Evec_retained' + repmat(meanDigit,nsamples,1); % reconstruction of x
figure,
subplot(1,2,1); 
imagesc(Dtrain); 
subplot(1,2,2);
imagesc(x_reconstruction);
end
