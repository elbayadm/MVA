import os.path as osp
import numpy as np
from argparse import ArgumentParser

from utils import read_list, write_list, pickle


def write_matrix(mat, file_path):
    content = [' '.join(map(str, r)) for r in mat]
    with open(file_path, 'w') as f:
        f.write('\n'.join(content))


def generate_matrix_q(noise_level):
    #Maintain 5% ratio of glasses
    nl0to1=noise_level
    nl1to0=.95/.05*nl0to1
    # nl1to0=noise_level/10
    q = np.array([[1-nl0to1,nl1to0],[nl0to1,1-nl1to0]],dtype=np.float)
    return q


def parse(file_path):
    lines = read_list(file_path)
    lines = map(str.split, lines)
    files, labels = zip(*lines)
    labels = map(int, labels)
    return (files, labels)


def corrupt(labels, q):
    n = len(q)
    cdf = np.cumsum(q, axis=0)
    noisy_labels = []
    for y in labels:
        r = np.random.rand()
        for k in xrange(n):
            if r <= cdf[k, y]:
                noisy_labels.append(k)
                break
    assert len(noisy_labels) == len(labels)
    return noisy_labels


def write_file_label_list(files, labels, file_path):
    content = ['{} {}'.format(f, l) for f, l in zip(files, labels)]
    write_list(content, file_path)


def main(args):
    q = generate_matrix_q(args.level)
    q=np.transpose(q)
    write_matrix(q, osp.join(args.data_root, 'matrix_q'+repr(args.level)+'.txt'))
    pickle(q, osp.join(args.data_root, 'matrix_q'+repr(args.level)+'.pkl'))
    # files, labels = parse(osp.join(args.data_root, 'clean.txt'))
    # noisy_labels = corrupt(labels, q)
    # write_file_label_list(files, noisy_labels,
    #     osp.join(args.data_root, 'noisy'+repr(args.level)+'.txt'))
    # write_list(noisy_labels, osp.join(args.data_root, 'noisy'+repr(args.level)+'_labels.txt'))

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Generate noisy labels of given level")
    parser.add_argument('data_root',
        help="Root directory containing train.txt")
    parser.add_argument('--level', type=float, default=0.5)
    parser.add_argument('--size', type=int, default=10000)
    args = parser.parse_args()
    main(args)