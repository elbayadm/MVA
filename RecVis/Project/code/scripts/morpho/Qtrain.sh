#!/usr/bin/env sh
cd $(dirname ${BASH_SOURCE[0]})/../../
caf=caffe/build/tools/caffe

#Change labelset in the net prototxt !!
NL=001

$caf train \
  -solver models/morpho/Qmorpho_solver.prototxt \
  -weights snaps/meso/morpho/NL${NL}/morpho_p1_iter_10000.caffemodel \
  2>&1 | tee logs/morpho/NL${NL}/multi.log

