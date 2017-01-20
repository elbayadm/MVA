function [y, inds] = vl_nnmaxout(x,groupSize,inds,dzdy)
% VL_NNMAXOUT CNN maxout layer
%   Y = VL_NNMAXOUT(X) applies maxout to the data X, taking the maximum 
%   over all channels.
%
%   [Y, inds] = VL_NNMAXOUT(X, groupSize) splits X channels into groups of
%   groupSize and returns the maximum response in each group. inds should be
%   a vector of indices to be passed as an argument in vl_nnmaxout during the
%   backpropagation step and its purpose is to determine which elements of
%   the derivative dz/dy will be used to compute dz/dx.
%
%   Y = VL_NNMAXOUT(X, groupSize, inds, dzdy) computes the
%   derivatives DZDX of the network relative to the input X given the 
%   the derivative DZDY relative to the outut Y and the indices
%   corresponding to maximum elements from the forward step.

sz = size(x);
if nargin < 2, groupSize = sz(3); end
forward = (nargin < 3);

if mod(sz(3),groupSize) ~= 0
    error('groupSize does not divide the number of channels')
end

nGroups = sz(3)/groupSize;

if forward  %==========================Forward-propagation

	y=zeros([sz(1:2),nGroups,sz(4)]);
	inds=zeros(size(x));
	for i=1:nGroups
		y(:,:,i,:)=max(x(:,:,(i-1)*groupSize+(1:groupSize),:),[],3);
		inds(:,:,(i-1)*groupSize+(1:groupSize),:)=(repmat(y(:,:,i,:),1,1,groupSize,1)==x(:,:,(i-1)*groupSize+(1:groupSize),:));
	end
	y=single(y);
	%fprintf('Check sum(inds)=%d\n',sum(inds(:)));


else %======================================Back-propagation

y=inds.*repelem(dzdy,1,1,groupSize,1);
end
