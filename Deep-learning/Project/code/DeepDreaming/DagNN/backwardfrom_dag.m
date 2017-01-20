function  backwardfrom_dag(net,derOutputs, ll)
% set the input values
if iscell(derOutputs)
for i=1:length(ll)
   net.vars(net.layers(ll(i)).outputIndexes).der = derOutputs{i} ;
end
else
    net.vars(net.layers(ll).outputIndexes).der = derOutputs;
end
derInputs = {} ;
derOutputs = {} ;
exe = net.executionOrder;
[~,l]=ismember(max(ll),exe);
for layer = net.layers(exe(l:-1:1))
  in = layer.inputIndexes ;
  out = layer.outputIndexes ;
  par = layer.paramIndexes ;
  inputs = {net.vars(in).value} ;
  derOutputs = {net.vars(out).der} ;
  [derInputs, ~] = layer.block.backward(inputs, {net.params(par).value}, derOutputs) ;
  % accumulate derivatives
      for i = 1:numel(in) 
        v = in(i) ;
        if net.numPendingVarRefs(v) == 0 || isempty(net.vars(v).der)
            net.vars(v).der = derInputs{i} ;
        elseif ~isempty(derInputs{i})
          net.vars(v).der = net.vars(v).der + derInputs{i} ;
        end
        net.numPendingVarRefs(v) = net.numPendingVarRefs(v) + 1 ;
      end
end