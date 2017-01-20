cd $(dirname ${BASH_SOURCE[0]})/../

#Draw the models
# python py/draw_net.py --rankdir TB models/morpho/Qmorpho_trainval.prototxt figures/Qmorpho.png
# python py/draw_net.py --rankdir TB models/morpho/meso/trainval.prototxt figures/morpho.png

# python py/draw_net.py --rankdir TB models/cifar10/Qcifar_trainval.prototxt figures/Qcifar.png
# python py/draw_net.py --rankdir TB models/cifar10/meso/trainval.prototxt figures/cifar.png

python py/draw_net.py --rankdir TB models/trainval.prototxt figures/bootstrap.png

