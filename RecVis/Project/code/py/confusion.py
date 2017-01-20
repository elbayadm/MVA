import os.path as osp
import sys
import numpy as np
from argparse import ArgumentParser
import caffe
import scipy.io as sio


def main(args):
	caffe.set_mode_cpu()
	net = caffe.Net('models/morpho/'+args.proto+'.prototxt', 
                'snaps/'+args.model+'/Q_iter_'+str(args.iter)+'.caffemodel', 
                caffe.TEST)
	print("blobs {}\nparams {}".format(net.blobs.keys(), net.params.keys()))
	Q=net.params['confusion'][0].data
	Q=np.asmatrix(Q)
	print(Q)
	sio.savemat('Confusion/'+args.model+'_'+str(args.iter)+'.mat',{'Q':Q})
	
if __name__ == '__main__':
    parser = ArgumentParser(
        description="Extract confusion matrix")
    parser.add_argument('--iter', type=int, default=2500)
    parser.add_argument('--model')
    parser.add_argument('--proto')
    args = parser.parse_args()
    main(args)