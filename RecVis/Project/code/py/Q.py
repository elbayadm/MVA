import os.path as osp
import numpy as np
from argparse import ArgumentParser
from utils import pickle
import caffe
def write_matrix(mat, file_path):
    content = [' '.join(map(str, r)) for r in mat]
    with open(file_path, 'w') as f:
        f.write('\n'.join(content))

fp = 0.95
fn = 0.95
tp = 0.05
tn = 0.05
data = np.array([[fn ,tn],[tp,fp]],dtype=np.float)
write_matrix(data,'../data/infogain/Q17.txt')
pickle(data,'../data/infogain/Q17.pkl')


shape = data.shape
shape = (1,) * (4 - len(shape)) + shape
data = data.reshape(shape)
blob = caffe.proto.caffe_pb2.BlobProto()
blob.num, blob.channels, blob.height, blob.width = data.shape
blob.data.extend(list(data.ravel().astype(float)))
with open('../data/infogain/Q17.binaryproto', 'wb') as f:
    f.write(blob.SerializeToString())