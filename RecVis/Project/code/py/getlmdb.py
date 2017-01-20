import caffe
import lmdb
import os
import caffe.proto.caffe_pb2
from caffe.io import datum_to_array
import scipy.io as sio
import numpy as np
lmdb_env = lmdb.open('../data/lmdbs/test')
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()
arr = []
labels =[]
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    print(label)
    labels.append(label)
    data = caffe.io.datum_to_array(datum)
    for d in data:
        print d
    	arr.append(np.array(data))
sio.savemat('test_images.mat',{'Images':arr, 'Labels' : labels})
