%% Settings:
clearvars;close all;clc;
set(0,'DefaultAxesFontSize',14)
set(0,'DefaultTextFontSize',14)
col=[
    9 125 18;
    97 6 158;
    199 2 45;
    2 2 214;
]/255;
tiny= 1e-10;

% Loading toolboxes:
addpath(genpath('matlab-toolboxes/toolbox_signal'))
addpath(genpath('matlab-toolboxes/toolbox_general'))

%% ---------------------------------------
%% Best M-terms Non-linear Approximation:
%% ---------------------------------------
n = 512;
f = rescale( load_image('lena', n) );
figure(1);clf;
imageplot(f);
%% Fourier approximation:
fF = fft2(f)/n;
imageplot(log(1e-5+abs(fftshift(fF))));


% Thresholding:
T = .3;
c = fF .* (abs(fF)>T);

% Inverse Fourier:
fM = real(ifft2(c)*n);
figure(1);clf;
imageplot(clamp(fM));

%% Exo 1:

Ms = round([1/100 1/20]*n^2 ); %[1/100 1/20]*N
figure(1);clf;
for i=1:length(Ms)
    M = Ms(i);
    fFsorted = sort(abs(fF(:)),'descend');
    T = fFsorted(M);
    fFT = fF .* (abs(fF)>=T);
    fM = real( ifft2(fFT)*n );
    imageplot( clamp(fM), ['M/N=' num2str(M/n^2,2) ', SNR=' num2str(snr(f,fM),3) 'dB'], 1,2,i);
end

%% Exo 2:
cR = sort(abs(fF(:)),'descend');
figure(1);clf;
plot(log10(cR),'lineWidth', 2); 
xlabel('index')
ylabel('coefficients magnitudes')
axis('tight');

%% Exo 3:
err_fft = max( norm(f(:))^2 - cumsum(cR.^2), tiny);  
figure(1);clf;
plot(log10(err_fft / norm(f(:))^2),'lineWidth', 2); 
ax = [1 n^2/50 -2.35 0];
axis(ax);
xlabel('M')
title('$\log_{10}(\epsilon^2[M]/\|f\|^2)$','interpreter','latex');

%% Wavelet Approximation:
Jmin = 1;
options.h = compute_wavelet_filter('Daubechies',10);
fW = perform_wavortho_transf(f,Jmin,+1, options);

figure(1);clf;
plot_wavelet(fW,Jmin);
title('Wavelet coefficients');

%% Exo 4:
figure(1);clf;
Ms = round([1/100 1/20]*n^2 ); %[1/100 1/20]*N
for i=1:length(Ms)
    M = Ms(i);
    fWsorted = sort(abs(fW(:)),'descend');
    T = fWsorted(M);
    fWT = fW .* (abs(fW)>=T);
    fM = perform_wavortho_transf(fWT,Jmin,-1, options);
    imageplot( clamp(fM), ['M/N=' num2str(M/n^2,2) ', SNR=' num2str(snr(f,fM),3) 'dB'], 1,2,i);
end

%% Exo 5:
cR = sort(abs(fW(:)),'descend');
err_wav = max( norm(f(:))^2 - cumsum(cR.^2), tiny);  
figure(1);clf;
plot(log10(err_wav / norm(f(:))^2),'lineWidth', 2,'color','b'); 
hold on,
plot(log10(err_fft / norm(f(:))^2),'lineWidth', 2,'color','r'); 
legend({'Wavelets' , 'Fourier'},'location','northeast')
ax = [1 n^2/50 -2.35 0];
axis(ax);
xlabel('M')
title('$\log_{10}(\epsilon^2[M]/\|f\|^2)$','interpreter','latex');

%% Cosine Approximation
fC = dct2(f);
figure(1);clf;
imageplot(log(1e-5+abs(fC)));

%% Exo 6:
Ms = round([1/100 1/20]*n^2 ); %[1/100 1/20]*N
figure(1);clf;
for i=1:length(Ms)
    M = Ms(i);
    fCsorted = sort(abs(fC(:)),'descend');
    T = fCsorted(M);
    fCT = fC .* (abs(fC)>=T);
    fM = idct2(fCT);
    imageplot( clamp(fM), ['M/N=' num2str(M/n^2,2) ', SNR=' num2str(snr(f,fM),3) 'dB'], 1,2,i);
end

%% Exo 7:
cR = sort(abs(fC(:)),'descend');
err_dct = max( norm(f(:))^2 - cumsum(cR.^2), tiny);  
figure(1);clf;
plot(log10(err_dct / norm(f(:))^2),'lineWidth', 2,'color','b'); 
hold on,
plot(log10(err_fft / norm(f(:))^2),'lineWidth', 2,'color','r'); 
legend({'DCT' , 'Fourier'},'location','northeast')
ax = [1 n^2/50 -2.35 0];
axis(ax);
xlabel('M')
title('$\log_{10}(\epsilon^2[M]/\|f\|^2)$','interpreter','latex');

%% Local Cosine Approximation
w = 16;
fL = zeros(n,n);
% Patch extraction:
i = 5;
j = 7;
seli = (i-1)*w+1:i*w;
selj = (j-1)*w+1:j*w;
P = f(seli,selj);
fL(seli,selj) = dct2(P);
figure(1);clf;
imageplot(P,'Patch',1,2,1);
imageplot(dct2(P-mean(P(:))),'DCT',1,2,2);

%% Exo 8:
fL = zeros(n,n);
nw = floor(n/w);
for i=1:nw
    for j=1:nw
        seli = (i-1)*w+1:i*w;
        selj = (j-1)*w+1:j*w;
        fL(seli,selj) = dct2( f(seli,selj) );
    end
end
figure(1);clf;
imageplot(min(abs(fL),.005*w*w));

%% Exo 9:
flc = fL;
for i=1:n/w
    for j=1:n/w
        seli = (i-1)*w+1:i*w;
        selj = (j-1)*w+1:j*w;
        flc(seli,selj) = idct2( flc(seli,selj) );
    end
end
fprintf('Error |f-flc|/|f| = %.4e\n',norm(f(:)-flc(:))/norm(f(:)));

%% Exo 10:
Ms = round([1/100 1/20]*n^2 );
figure(1); clf;
for k = 1:length(Ms)
    M = Ms(k);
    fLsorted = sort(abs(fL(:)),'descend');
    T = fLsorted(M);
    fLC = fL .* (abs(fL)>=T);
    fM = fLC;
    for i=1:n/w
        for j=1:n/w
            seli = (i-1)*w+1:i*w;
            selj = (j-1)*w+1:j*w;
            fM(seli,selj) = idct2( fM(seli,selj) );
        end
    end
    % display
    imageplot( clamp(fM), ['M/N=' num2str(M/n^2,2) ', SNR=' num2str(snr(f,fM),3) 'dB'], 1,2,k);
end

%% Exo 11:
cR = sort(abs(fLC(:)),'descend');
err_ldct = max( norm(f(:))^2 - cumsum(cR.^2), tiny);  
figure(1);clf;
plot(log10(err_ldct / norm(f(:))^2),'lineWidth', 2,'color','b');
hold on,
plot(log10(err_dct / norm(f(:))^2),'lineWidth', 2,'color','g'); 
hold on,
plot(log10(err_fft / norm(f(:))^2),'lineWidth', 2,'color','r'); 
hold on,
plot(log10(err_wav / norm(f(:))^2),'lineWidth', 2,'color','k'); 

legend({'local-DCT','DCT' , 'Fourier','Wavelets'},'location','northeast')
ax = [1 n^2/50 -2.35 0];
axis(ax);
xlabel('M')
title('$\log_{10}(\epsilon^2[M]/\|f\|^2)$','interpreter','latex');

%% Comparison of Wavelet Approximations of Several Images
n = 512;
fList(:,:,1) = rescale( load_image('regular3',n) );
fList(:,:,2) = rescale( load_image('phantom',n) );
fList(:,:,3) = rescale( load_image('lena',n) );
fList(:,:,4) = rescale( load_image('mandrill',n) );
figure(1);clf;
for i=1:4
    imageplot(fList(:,:,i),'', 2,2,i);
end
%% Exo 12:
Jmin = 1;
options.h = compute_wavelet_filter('Daubechies',10);
figure(1);clf;
err = [];
Ms = 10:n*n;
for i=1:size(fList,3)
    f = fList(:,:,i);
    fW = perform_wavortho_transf(f,Jmin,+1, options);
    cR = sort(abs(fW(:)),'descend');
    e = max( norm(f(:))^2 - cumsum(cR.^2), tiny);  
    plot(log10(Ms), log10(e(Ms)/e(Ms(1))),'linewidth',2,'color',col(i,:));
    hold on,
end
xlabel('log_{10}(M)')
ylabel('log_{10}(f-f_M)')
title('log_{10}( \epsilon^2[M]/||f||^2 )');
legend({'regular', 'phantom', 'lena', 'mandrill'},'location','southwest');
axis([1  4.5 -7 0])