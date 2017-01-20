function [loss]=barrier_loss(Q,p,A,b,t,v)
loss=t*(v'*Q*v+p'*v)-sum(log(b-A*v));