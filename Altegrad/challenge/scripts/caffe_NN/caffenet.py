import caffe
import h5py
import os
import numpy as np
import sys

def format_data(X,Y):
	data = {}
	data['input'] = np.reshape(X,(X.shape[0], 1, 1, X.shape[1]))
	data['output'] = Y
	return data

def save_hdf5(filename, data):
    '''
    HDF5 is one of the data formats Caffe accepts
    '''
    with h5py.File(filename, 'w') as f:
        f['data'] = data['input'].astype(np.float32)
        f['label'] = data['output'].astype(np.float32)

def train(solver_proto):
    '''
    Train the ANN
    '''
    caffe.set_mode_cpu()
    solver = caffe.get_solver(solver_proto)
    solver.solve()
    
def nn_single_predict(deploy_proto, caffemodel_filename, input, net = None):
    '''
    Get the predicted output, i.e. perform a forward pass
    '''
    if net is None:
        net = caffe.Net(deploy_proto,caffemodel_filename, caffe.TEST)
        
    out = net.forward(data=input)
    return int(np.argmax(out[net.outputs[0]], axis=1))

def nn_predict(deploy_proto, caffemodel_filename, inputs):
    '''
    Get several predicted outputs
    '''
    outputs = []
    net = caffe.Net(deploy_proto,caffemodel_filename, caffe.TEST)
    for input in inputs:
        #print(input)
        outputs.append(nn_single_predict(deploy_proto, caffemodel_filename, input, net))
    return outputs    

def citations_network(Xtrain,Ytrain,Xtest,Ytest):
	# params:
	solver_proto = "caffe_NN/solver.prototxt"
	train_filename = "caffe_NN/citations_train.hdf5"
	test_filename = "caffe_NN/citations_test.hdf5"
	train_data = format_data(Xtrain,Ytrain)
	test_data = format_data(Xtest,Ytest)
	save_hdf5(train_filename, train_data)
	save_hdf5(test_filename, test_data)
	train(solver_proto)

def citations_predict(Xtest):
	deploy_proto = "caffe_NN/deploy.prototxt"
	caffemodel_filename = 'caffe_NN/citations_iter_100000.caffemodel'
	inputs = np.reshape(Xtest,(Xtest.shape[0], 1, 1, Xtest.shape[1]))
	preds = nn_predict(deploy_proto, caffemodel_filename, inputs)
	return preds