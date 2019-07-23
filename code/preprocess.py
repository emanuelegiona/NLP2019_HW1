from argparse import ArgumentParser
import os
import unicodedata


# Writes up to BATCH_SIZE lines at once to input and label files
BATCH_SIZE = 128


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("dataset_path", help="The path of the plain dataset file")
    parser.add_argument("dir_output_path", help="The path of the output directory")

    return parser.parse_args()


def unicode_normalization(word):
    """
    Normalizes a word, or parts of it.
    For each character, if its unicode name contains "FULLWIDTH", performs normalization.
    :param word: Word to be normalized
    :return: normalized word
    """
    normalized_word = []

    for i in range(len(word)):
        # catch trailing '\n' in words and preserve them
        try:
            unicode_name = unicodedata.name(word[i])
        except ValueError:
            #normalized_word.append("\n")
            continue

        if unicode_name.find("FULLWIDTH") >= 0:
            normalized_word.append(
                unicodedata.normalize('NFKC', word[i])
            )
        else:
            normalized_word.append(word[i])

    return ''.join(normalized_word)


def parse_line(line):
    """
    Performs the actual pre-processing on a given line of the dataset:
        - converting digits, punctuation, and latin letters to half-width UTF-8
        - building input (no whitespaces left) and label (BIES-formatted) files
    :param line: Line taken from the dataset
    :return: tuple (whitespace-stripped version of the line itself, BIES-formatted string corresponding to the line itself)
    """
    space = " "
    if line.find('\u3000') >= 0:
        space = '\u3000'

    line = line.split(space)
    normalized_line = []

    labels = []
    for word in line:
        word = unicode_normalization(word)
        normalized_line.append(word)

        if len(word) == 1:
            labels.append("S")
            continue

        for i in range(len(word)):
            if i == 0:
                labels.append("B")
            elif i == len(word) - 1:
                labels.append("E")
            else:
                labels.append('I')

    return ''.join(normalized_line), ''.join(labels)


def pre_process(dataset_path, dir_output_path):
    """
    Performs some pre-processing routines on the given dataset.
    The given dataset is assumed to already be in simplified Chinese and stored in UTF-8 encoding.

    :param dataset_path: The path of the plain dataset file
    :param dir_output_path: The path of the output directory
    :return: None
    """
    print("Starting pre-processing...")

    dataset_name = os.path.splitext(os.path.basename(dataset_path))

    input_path = dataset_name[0] + "_input" + dataset_name[1]
    input_path = os.path.join(dir_output_path, input_path)

    label_path = dataset_name[0] + "_label" + dataset_name[1]
    label_path = os.path.join(dir_output_path, label_path)

    # holds a batch of whitespace-stripped lines
    inputs_batch = []

    # holds a batch of BIES-formatted labels for the inputs batch
    labels_batch = []

    # Overwrite with fresh files, if already existing
    with \
            open(input_path, mode='w', encoding='utf-8'), \
            open(label_path, mode='w', encoding='utf-8'):
        pass

    with \
            open(dataset_path, encoding='utf-8') as dataset,\
            open(input_path, mode='a', encoding='utf-8') as input_file,\
            open(label_path, mode='a', encoding='utf-8') as label_file:
        for line in dataset:
            (input_line, label_line) = parse_line(line)
            inputs_batch.append(input_line)
            labels_batch.append(label_line)

            # BATCH_SIZE reached
            if len(inputs_batch) == BATCH_SIZE:
                input_file.writelines("%s\n" % w for w in inputs_batch)
                label_file.writelines("%s\n" % l for l in labels_batch)

                inputs_batch = []
                labels_batch = []

        # incomplete batch
        if len(inputs_batch) > 0:
            input_file.writelines("%s\n" % w for w in inputs_batch)
            label_file.writelines("%s\n" % l for l in labels_batch)

            del inputs_batch
            del labels_batch

    print("Pre-processing of file %s is over." % (dataset_name[0] + dataset_name[1]))


if __name__ == '__main__':
    args = parse_args()
    pre_process(args.dataset_path, args.dir_output_path)

