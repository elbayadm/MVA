function [emp]= emp_hess(Q,p,A,b,t,v)
emp=[];
step=.0001;
for i=1:size(v,1)
    e=zeros(size(v));
    e(i)=step;
    emp=[emp (emp_grad(Q,p,A,b,t,v+e)-emp_grad(Q,p,A,b,t,v-e))/2/step];
end