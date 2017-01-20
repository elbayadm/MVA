cd $(dirname ${BASH_SOURCE[0]})/../../

NL=${1}

ext=caffe/build/tools/extract_features_txt

# 01 / 02 / 03

# the Q matrix
python2 py/extract_weights_morphoB.py --nl ${NL}

#Meso logs (GPU)
python py/parse_log.py logs/morpho/NLB${NL}/p1.log logs/morpho/NLB${NL}/ --delimiter ';'

#Local log - Q finetuning
python py/parse_log.py logs/morpho/NLB${NL}/p2.log logs/morpho/NLB${NL}/ --delimiter ';'

# Extract features:
mkdir -p features/morpho/NLB${NL}

$ext snaps/morpho_NLB${NL}/Q_iter_2500.caffemodel models/morpho/Qmorpho_ext.prototxt confusion features/morpho/NLB${NL}/conf_2500.txt 4661

$ext snaps/morpho_NLB${NL}/Q_iter_2500.caffemodel models/morpho/Qmorpho_ext.prototxt prob features/morpho/NLB${NL}/prob_2500.txt 4661

$ext snaps/meso/morpho/NLB${NL}/morpho_p1_iter_10000.caffemodel models/morpho/ext.prototxt prob features/morpho/NLB${NL}/prob_10000.txt 4661
