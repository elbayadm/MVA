do_print=1;
colors=hsv(12);
%% vertical & horizontal size of lattice
sv = 4;
sh = 4;
% connection strength (>0 favors similar neighboring spins)
J = .5;
A= zeros(sv*sh,sv*sh);
% connect only grid-based neighbors
for h = 1:sh
    for v= 1:sv
        neighs =[];
        pt = (h-1)*sv + v;
        if v>1
            neighs = [neighs, (h-1)*sv + v-1];
        end
        if v<sv
            neighs = [neighs, (h-1)*sv + v+1];
        end
        if h>1
            neighs = [neighs, (h-2)*sv + v];
        end
        if h<sh
            neighs = [neighs, (h)*sv + v];
        end
        A(pt,neighs) = J;
        A(neighs,pt) = J;
    end
end

%% Exhaustive enumeration of possible states: 2^{sv*sh}
nnodes=sv*sh;
V=[-1 1];
states = get_all_states(nnodes,V) ;       

%% Measuring the energy of each state:

C = num2cell(states,2);  
energies=cellfun(@(x) -1/2*x*A*x',C);

%% compute partition function
Z = sum(exp(-energies));

%% compute probability of all states
P=1/Z*exp(-energies);
if do_print
    figure(1)
    plot(P);
    xlabel('state','fontsize',20);
    ylabel('P(state)','fontsize',20);
    print(1,'-dpdf','images/tocrop/Boltzmann_pstates.pdf', '-opengl');

    [~,i_max]= max(energies);
    [~,i_min]= min(energies);
    figure(2)
    subplot(1,2,1);
    imshow(reshape(states(i_max,:),[sv,sh]),[-1,1])
    t=title('max-energy state');
    set(t,'FontSize',14, 'FontWeigh','normal');
    subplot(1,2,2);
    imshow(reshape(states(i_min,:),[sv,sh]),[-1,1])
    t=title('min-energy state');
    set(t,'FontSize',14, 'FontWeigh','normal');
    print(2,'-dpdf','images/tocrop/Boltzmann_minmax.pdf', '-opengl');
end

%% Brute force features expectation
[is,js]     = find(A);
nedges      = length(is);
E_ij=zeros(nedges,1);
for k=1:nedges
  E_ij(k) = sum(P.*states(:,is(k)).*states(:,js(k)));   
end
if do_print
    figure(3);
    plot(E_ij,'linewidth',2);
    xlabel('feature index ~ edge','fontsize',20);
    ylabel('Expectation','fontsize',20);
    print(3,'-dpdf','images/tocrop/Boltzmann_expect.pdf', '-opengl');
end

%% Gibbs sampling
% total number of Gibbs sampling iterations
Nsamples  = 4000;
sample=zeros(Nsamples,nnodes);
% consecutive samples are not independent -use every 100th sample
% for estimation
step_sample = 100;
samples_tot = (Nsamples-1)*step_sample;

% running estimate of the expectation
avg_ij      = zeros(Nsamples,nedges);
% current sum of state products
sum_ij      = zeros(nedges,1);


fprintf(1,'Gathering samples .. \n');
rng(1,'V4');% to ensure we get reproducible results

% randomly initialize network's state
X = 2*double(rand(nnodes,1)>.5)-1;
sample(1,:)=X;

% number of terms that have been considered so far
count = 1;
for it = 1:samples_tot
    % select a pixel at random
    ix = ceil( sh * rand(1) );
    iy = ceil( sv * rand(1) );
    % find its index in 2D array
    position = iy + sv*(ix-1);
    % edge weights connecting 'position' to all other nodes
    neighsCon = A(position,:);
    % Gibbs sampling, inner loop
        % energy if we assign state +1 to 'position' (+ a constant)
        ener_p1 =neighsCon*X;
        % ditto for state -1 (+ same constant)
        ener_m1 =-ener_p1;  
        % posterior probability that 'position' will be +1 (constant
        % absorbed)
        p1 = 1/(1+exp(2*ener_m1));
        % decide whether to set 'position' to +1 or -1
        state = 2*double(rand(1)<p1)-1;
     %update X
        X(position) = state;
        
     if ~mod(it-1,step_sample)
         count=count+1;
         sample(count,:)=X;
         %Update statistics:
         for k=1:nedges
              sum_ij(k)=sum_ij(k)+X(is(k))*X(js(k));
              avg_ij(count,k)= sum_ij(k)/count;
         end
     end
end

fprintf('Done sampling \n');
%%
if do_print
    figure(4)
    xlabel('feature index ~ edge','fontsize',20);
    ylabel('Expectation','fontsize',20);
    plot(E_ij,'color',colors(2,:),'linewidth',2);
    hold on,
    plot(avg_ij(end,:),'color',colors(4,:),'linewidth',2);
    set(gca,'ylim',[0 1]);
    legend({'Brute Force','Monte Carlo'})
    print(4,'-dpdf','images/tocrop/gibbs_1.pdf', '-opengl');
end
%%  showing intermediate expectations
if do_print   
    shown    = ceil(linspace(1,count,11));
    shown(1) = [];
    nshown = length(shown);
    figure(5)
    for k=1:nshown
        str{k}  = sprintf('Iteration: %i\n',shown(k));
        plot(avg_ij(shown(k),:),'color',colors(k,:),'linewidth',2);
        hold on
    end
    str{nshown+1} = 'Brute Force';
    plot(E_ij,'k','linewidth',2);
    axis([0,length(E_ij),0,1])
    legend(str,'location','southwest');
    print(5,'-dpdf','images/tocrop/gibbs_iter.pdf', '-opengl');
end

%% Parameter estimation
it_train  = 10;
J_        = .5; 
t=load('E_ij');
E_ij = t.E_ij;
J_track=zeros(1,it_train+1);
Stats=zeros(it_train+1,nedges);

for it_estim=1:(it_train+1)
    A(A~=0) = J_;
    rng(1,'V4');% to ensure we get reproducible results
    % randomly initialize network's state
    X = 2*double(rand(nnodes,1)>.5)-1;
    sample(1,:)=X;
    % running estimate of the expectation
    avg_ij      = zeros(Nsamples,nedges);
    % current sum of state products
    sum_ij      = zeros(nedges,1);

    % number of terms that have been considered so far
    count = 1;
    for it = 1:samples_tot
        % select a pixel at random
        ix = ceil( sh * rand(1) );
        iy = ceil( sv * rand(1) );
        % find its index in 2D array
        position = iy + sv*(ix-1);
        % edge weights connecting 'position' to all other nodes
        neighsCon = A(position,:);
        % Gibbs sampling, inner loop
            % energy if we assign state +1 to 'position' (+ a constant)
            ener_p1 =neighsCon*X;
            % ditto for state -1 (+ same constant)
            ener_m1 =-ener_p1;  
            % posterior probability that 'position' will be +1 (constant
            % absorbed)
            p1 = 1/(1+exp(2*ener_m1));
            % decide whether to set 'position' to +1 or -1
            state = 2*double(rand(1)<p1)-1;
         %update X
            X(position) = state;

         if ~mod(it-1,step_sample)
             count=count+1;
             sample(count,:)=X;
             %Update statistics:
             for k=1:nedges
                  sum_ij(k)=sum_ij(k)+X(is(k))*X(js(k));
                  avg_ij(count,k)= sum_ij(k)/count;
             end
         end
    end
    fprintf('Done sampling -updating parameters\n');
    avg_final = avg_ij(end,:);
    Stats(it_estim,:) = avg_final;
    J_  = J_ - .015*(sum(avg_final - E_ij));
    J_track(it_estim) = J_;
end

%% Showing results
if do_print
    nshown = 10;    
    figure(6)
    for k=1:(it_train+1)
        str{k}  = sprintf('Iteration %d',k);
        plot(Stats(k,:),'color',colors(k,:),'linewidth',2);
        hold on
    end
    plot(E_ij,'k','linewidth',2);
    axis([0,length(E_ij),0,1]);
    xlabel('Edge','fontsize',20);
    ylabel('Expectation','fontsize',20);
    str{k+1}='Desired';
    legend(str,'location','southwest');
    print(6,'-dpdf','images/tocrop/estim_updates.pdf', '-opengl');    
    
    figure(7)
    plot(Stats(end,:),'color',colors(it_train+1,:),'linewidth',2);
    hold on,
    
    plot(E_ij,'k','linewidth',2);
    legend({'with estimated model','desired'})
    axis([0,length(E_ij),0,1]);
    xlabel('Edge','fontsize',20);
    ylabel('Expectation','fontsize',20);
    print(7,'-dpdf','images/tocrop/estim_final.pdf', '-opengl');  
    
    figure(8)
    plot(J_track);
    xlabel('Iteration','fontsize',20);
    ylabel('estimated (J)','fontsize',20);
    print(8,'-dpdf','images/tocrop/Jtrack.pdf', '-opengl');  
end