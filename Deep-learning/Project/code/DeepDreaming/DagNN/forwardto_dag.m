function  forwardto_dag(net, inputs, ll)
% Forward in the net up to layer 'll'
% set the input values
[net.vars(1).value] = inputs ;
inputs = [] ;
exe = net.executionOrder;
[~,l]=ismember(ll,exe);
net.numPendingVarRefs = [net.vars.fanout] ;
for layer = net.layers(exe(1:l))
  in = layer.inputIndexes ;
  out = layer.outputIndexes ;
  par = layer.paramIndexes ;
  inputs = {net.vars(in).value} ;
  for v=in
      net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1 ;
  end

  outputs = layer.block.forward(inputs, {net.params(par).value}) ;
  [net.vars(out).value] = deal(outputs{:}) ;
end

