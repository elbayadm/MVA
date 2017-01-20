cd $(dirname ${BASH_SOURCE[0]})/../../

NL=${1}
ext=caffe/build/tools/extract_features_txt

# 001 / 002 / 003

# the Q matrix
python2 py/extract_weights_morpho.py --nl ${NL}

#Meso logs (GPU)
python py/parse_log.py logs/morpho/NL${NL}/p1.log logs/morpho/NL${NL}/ --delimiter ';'

#Local log - Q finetuning
python py/parse_log.py logs/morpho/NL${NL}/p2.log logs/morpho/NL${NL}/ --delimiter ';'

#Extract features:
mkdir -p features/morpho/NL${NL}

$ext snaps/morpho_NL${NL}/Q_iter_2500.caffemodel models/morpho/Qmorpho_ext.prototxt confusion features/morpho/NL${NL}/conf_2500.txt 4661

$ext snaps/morpho_NL${NL}/Q_iter_2500.caffemodel models/morpho/Qmorpho_ext.prototxt prob features/morpho/NL${NL}/prob_2500.txt 4661

$ext snaps/meso/morpho/NL${NL}/morpho_p1_iter_10000.caffemodel models/morpho/ext.prototxt prob features/morpho/NL${NL}/prob_10000.txt 4661
