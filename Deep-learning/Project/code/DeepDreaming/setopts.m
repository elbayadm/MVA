function opts=setopts()
opts=[];

opts.iters = 10;
opts.step = 1.5;
opts.layer= 3; 

% Regularization
opts.decay=0; %L2 decay
opts.clip=0; % Minimal percentage pixel norm tolerated

% Jittering and filtering:
opts.jitter=0;
opts.blur=0; % Gaussian blur
opts.average=0; % Average blurring
opts.freq=6; % blurring frequency

%Iterative zooming
opts.octave=0;
opts.crop=0;
opts.scale=1.4;

% For multi channel cropping
opts.grid=5;
opts.strip=3;

% Objective function
opts.objective='N2'; % 'neuron' , 'sumN2' , 'guide'
% For single unit maximization || 'neuron'
opts.activate=1; 
% For multiple layers dreaming
opts.combine = [.5 .5];
% guiding image for controlling dreams  || 'guide'
% opts.guide='flower.jpg' 


% Printing and saving the outputs
opts.doviz = 1;
opts.doprint = 0;
opts.fig='';
opts.net='';

end