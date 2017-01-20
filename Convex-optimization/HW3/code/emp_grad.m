function [emp]= emp_grad(Q,p,A,b,t,v)
emp=zeros(size(v));
step=.0001;
for i=1:size(v,1)
    e=zeros(size(v));
    e(i)=step;
    emp(i)=(barrier_loss(Q,p,A,b,t,v+e)-barrier_loss(Q,p,A,b,t,v-e))/2/step;
end