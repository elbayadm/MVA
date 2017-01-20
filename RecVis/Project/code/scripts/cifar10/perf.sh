cd $(dirname ${BASH_SOURCE[0]})/../../

NL=${1}
ext=caffe/build/tools/extract_features_txt


# the Q matrix
python py/extract_weights.py --nl ${NL}

#Meso logs (GPU)
python py/parse_log.py logs/cifar10/NL0${NL}/f1.log logs/cifar10/NL0${NL}/ --delimiter ';'
python py/parse_log.py logs/cifar10/NL0${NL}/f2.log logs/cifar10/NL0${NL}/ --delimiter ';'
python py/parse_log.py logs/cifar10/NL0${NL}/f3.log logs/cifar10/NL0${NL}/ --delimiter ';'

#Local log - Q finetuning
python py/parse_log.py logs/cifar10/Qcifar10_NL0${NL}.log logs/cifar10/ --delimiter ';'

# Extract features:
mkdir -p features/cifar10/NL0${NL}

$ext snaps/cifar_NL0${NL}/Q_iter_5000.caffemodel models/cifar10/Qcifar_ext.prototxt confusion features/cifar10/NL0${NL}/conf_5000.txt 10000

$ext snaps/cifar_NL0${NL}/Q_iter_5000.caffemodel models/cifar10/Qcifar_ext.prototxt prob features/cifar10/NL0${NL}/prob_5000.txt 10000

$ext snaps/meso/cifar10/NL0${NL}/cifar10_full_iter_70000.caffemodel models/cifar10/meso/ext.prototxt prob features/cifar10/NL0${NL}/prob_70000.txt 10000