#!/usr/bin/env sh
cd $(dirname ${BASH_SOURCE[0]})/../../
caf=caffe/build/tools/caffe

#Change labelset in the net prototxt and the snapshot prefix !!
NL=6

$caf train \
  -solver models/cifar10/Qcifar_full_solver.prototxt \
  -weights snaps/meso/cifar10/NL0${NL}/cifar10_full_iter_70000.caffemodel \
  2>&1 | tee logs/cifar10/Qcifar10_NL0${NL}.log

