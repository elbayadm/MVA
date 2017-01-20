cd $(dirname ${BASH_SOURCE[0]})/../../

ext=caffe/build/tools/extract_features_txt
NL=001
MODEL=IP
PROTO=trainval_decay

#Confusion matrix
python2 py/confusion.py --model ${MODEL}_${NL} --proto ${PROTO}

python py/parse_log.py logs/morpho/NL${NL}/${MODEL}.log logs/morpho/NL${NL}/ --delimiter ';'

mkdir -p features/morpho/NL${NL}/${MODEL}

snap=snaps/${MODEL}_${NL}/Q_iter_2500.caffemodel

$ext ${snap} models/morpho/${PROTO}_ext.prototxt confusion features/morpho/NL${NL}/${MODEL}/conf_2500.txt 4661

$ext snaps/morpho_NL${NL}/Q_iter_2500.caffemodel models/morpho/${PROTO}_ext.prototxt prob features/morpho/NL${NL}/${MODEL}/prob_2500.txt 4661




