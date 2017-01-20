function res = backwardfrom_nn(net, n, dzdy, res, varargin)
opts.sync = false ;
opts.disableDropout = false ;
opts.cudnn = true ;
opts = vl_argparse(opts, varargin);

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end
gpuMode = isa(res(1).x, 'gpuArray') ;

res(n+1).dzdx = dzdy ;


for i=n:-1:1
l = net.layers{i} ;
res(i).backwardTime = tic ;
switch l.type
  case 'conv'
    if isfield(l,'weights')
         res(i).dzdx =  ...
            vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i+1).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
    else
        res(i).dzdx = ...
         vl_nnconv(res(i).x, l.filters, l.biases, ...
                res(i+1).dzdx, ...
                'pad', l.pad, 'stride', l.stride, ...
                cudnn{:}) ;
    end
  case 'pool'
    res(i).dzdx =vl_nnpool(res(i).x, l.pool, res(i+1).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                cudnn{:}) ;
  case 'normalize'
    res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i+1).dzdx) ;
  case 'softmax'
    res(i).dzdx = vl_nnsoftmax(res(i).x, res(i+1).dzdx) ;
  case 'loss'
    res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i+1).dzdx) ;
  case 'softmaxloss'
    res(i).dzdx =  vl_nnsoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
  case 'relu'
    if isfield(l, 'leak'), leak = {'leak', l.leak} ; else leak = {} ; end
        if ~isempty(res(i).x)
          res(i).dzdx =  vl_nnrelu(res(i).x, res(i+1).dzdx, leak{:}) ;
        else 
          % if res(i).x is empty, it has been optimized away, so we use this
          % hack (which works only for ReLU):
          res(i).dzdx = vl_nnrelu(res(i+1).x, res(i+1).dzdx, leak{:}) ;
        end
  case 'noffset'
    res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i+1).dzdx) ;
  case 'dropout'
    if opts.disableDropout
      res(i).dzdx = res(i+1).dzdx ;
    else
      res(i).dzdx = vl_nndropout(res(i).x, res(i+1).dzdx, 'mask', res(i+1).aux) ;
    end
  case 'custom'
    res(i) = l.backward(l, res(i), res(i+1)) ;
end
if gpuMode && opts.sync
  wait(gpuDevice) ;
end
res(i).backwardTime = toc(res(i).backwardTime) ;
end