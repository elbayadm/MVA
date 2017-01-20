cd $(dirname ${BASH_SOURCE[0]})/../

#NL=001
#LOG=HP1
FD=${1}
LOG=${2}

python py/parse_log.py logs/morpho/${FD}/${LOG}.log logs/morpho/${FD}/ --delimiter ';'
