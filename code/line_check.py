from argparse import ArgumentParser
import numpy as np


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_file_path", help="Input file path (no spaces)")
    parser.add_argument("label_file_path", help="Label file path (BIES format)")

    return parser.parse_args()


def line_check(input_file_path, label_file_path):
    """
    Checks for different length lines in input, label files
    :param input_file_path: Input file path (no spaces)
    :param label_file_path: Label file path (BIES format)
    :return:
    """
    lines1 = []
    lines2 = []

    with open(input_file_path, encoding='utf-8') as f1:
        for line in f1:
            line = line.strip()
            lines1.append(len(line))

    with open(label_file_path, encoding='utf-8') as f2:
        for line in f2:
            line = line.strip()
            lines2.append(len(line))

    lines1 = np.array(lines1)
    lines2 = np.array(lines2)

    diff = lines1-lines2

    print("There are %d different length lines." % np.count_nonzero(diff))


if __name__ == '__main__':
    args = parse_args()
    line_check(args.input_file_path, args.label_file_path)

