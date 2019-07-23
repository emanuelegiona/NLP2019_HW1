import math
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from math import ceil

# Writes up to BATCH_SIZE_FILE lines at once to files
BATCH_SIZE_FILE = 128


def split_to_words(line, word_size=0):
    """
    Auxiliary function to split lines into n-grams.
    :param line: Line read from a file
    :param word_size: 0 for words (default), 1 for char-unigrams, 2 for char-bigrams, and so on
    :return: List of words split according to word_size
    """
    words = []
    if word_size == 0:
        words = line.split()
    else:
        for i in range(len(line) - word_size + 1):
            word = line[i:i+word_size]
            words.append(word)

    return words

# --- Related to pre-trained embeddings ---


def make_vocab(dataset_path, vocab_path, vocab_size, word_size=0):
    """
    Creates a vocabulary of the given size from the given dataset, sorting words by frequency.
    :param dataset_path: File path of the dataset to build the vocabulary of
    :param vocab_path: File path where to export the vocabulary to
    :param vocab_size: Size of the vocabulary to create
    :param word_size: 0 for words (default), 1 for char-unigrams, 2 for char-bigrams, and so on
    :return: None
    """
    vocab_batch = []
    occurrences = {}

    # Overwrite with fresh file, if already existing
    with open(vocab_path, mode='w', encoding='utf-8'):
        pass

    with \
            open(dataset_path, encoding='utf-8') as dataset,\
            open(vocab_path, mode='a', encoding='utf-8') as output:
        for line in dataset:
            line = line.strip()
            words = split_to_words(line, word_size)
            for w in words:
                occurrences[w] = occurrences.get(w, 0) + 1

        print("Total Size: ", len(occurrences))

        # sort by decreasing occurrence number to only get the most frequent words
        for w in sorted(occurrences.items(), key=lambda x: x[1], reverse=True):
            # take the word from the tuple
            w = w[0]

            # stop once the vocabulary size is reached
            if vocab_size == 0:
                break

            vocab_size -= 1
            vocab_batch.append(w)

            # BATCH_SIZE reached
            if len(vocab_batch) == BATCH_SIZE_FILE:
                output.writelines("%s\n" % v for v in vocab_batch)
                vocab_batch = []

        del occurrences

        # incomplete batch
        if len(vocab_batch) > 0:
            output.writelines("%s\n" % v for v in vocab_batch)
            del vocab_batch


def word2vec_batch_generator(dataset, ngram_size, batch_size, window_size, word_to_idx):
    """
    Generator for Word2Vec batches.
    :param dataset: File handle of the dataset to train embeddings on
    :param ngram_size: 0 for words, 1 for char-unigrams, 2 for char-bigrams, and so on
    :param batch_size: Size of the batch for Word2Vec training
    :param window_size: Window size for Word2Vec training
    :param word_to_idx: Dictionary for word indexing
    :return: (batch inputs, batch labels) in a generator fashion
    """
    batch = np.zeros((batch_size,), dtype=np.int32)
    labels = np.zeros((batch_size, 1), dtype=np.int32)
    curr_batch_size = 0

    for sentence in dataset:
        sentence = sentence.strip()
        sentence = split_to_words(sentence, ngram_size)
        sentence = ['<S>'] + sentence + ['</S>']
        sentence = [word_to_idx[w] if w in word_to_idx else word_to_idx['<UNK>'] for w in sentence]

        sentence_size = len(sentence)
        for label_idx in range(sentence_size):
            min_idx = max(0, label_idx - window_size)
            max_idx = min(sentence_size, label_idx + window_size)

            window_idxs = [x for x in range(min_idx, max_idx) if x != label_idx]

            for input_idx in window_idxs:
                labels[curr_batch_size, 0] = sentence[input_idx]
                batch[curr_batch_size] = sentence[label_idx]
                curr_batch_size += 1

                if curr_batch_size == batch_size:
                    batch, labels = shuffle(batch, labels)
                    yield batch, labels

                    batch = np.zeros((batch_size,), dtype=np.int32)
                    labels = np.zeros((batch_size, 1), dtype=np.int32)
                    curr_batch_size = 0

    if curr_batch_size > 0:
        batch, labels = shuffle(batch, labels)
        yield batch[:curr_batch_size], labels[:curr_batch_size]


def train_embeddings(dataset_path, vocab_path, embeddings_path, embedding_size, ngram_size=0):
    """
    Trains embeddings on the given dataset via Word2Vec.
    :param dataset_path: File path of the dataset to train the embeddings on
    :param embeddings_path: File path where to export the embeddings to
    :param vocab_path: File path containing the vocabulary to use
    :param embedding_size: Embedding size
    :param ngram_size: 0 for words (default), 1 for char-unigrams, 2 for char-bigrams, and so on
    :return: None
    """
    word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<S>': 2, '</S>': 3}

    print("Reading vocabulary...")
    with open(vocab_path, encoding='utf-8') as vocab_file:
        for line in vocab_file:
            line = line.strip()
            if line not in word_to_idx:
                word_to_idx[line] = len(word_to_idx)

    idx_to_words = dict(zip(word_to_idx.values(), word_to_idx.keys()))

    # hyper-parameters
    BATCH_SIZE_W2V = 128
    WINDOW_SIZE_W2V = 1
    NEG_SAMPLES_W2V = 32
    NUM_EPOCHS_W2V = 100

    print("Creating model...")
    with tf.variable_scope("inputs"):
        inputs = tf.placeholder(tf.int32, shape=[None])
        labels = tf.placeholder(tf.int32, shape=[None, 1])

    with tf.variable_scope("embeddings"):
        embeddings = tf.get_variable(
            name="embeddings",
            initializer=tf.random_uniform(
                [len(word_to_idx), embedding_size],
                -1.0,
                1.0
            ),
            dtype=tf.float32
        )

        lookups = tf.nn.embedding_lookup(
            embeddings,
            inputs
        )

    with tf.variable_scope("weights"):
        nce_weights = tf.get_variable(
            name="weights",
            initializer=tf.truncated_normal(
                [len(word_to_idx), embedding_size],
                stddev=1.0/math.sqrt(embedding_size)
            )
        )

    with tf.variable_scope("bias"):
        nce_bias = tf.get_variable(
            name="bias",
            initializer=tf.zeros(len(word_to_idx))
        )

    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_bias,
                labels=labels,
                inputs=lookups,
                num_sampled=NEG_SAMPLES_W2V,
                num_classes=len(word_to_idx)
            )
        )

    with tf.variable_scope("optimizer"):
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.15).minimize(loss)

    print("Training embeddings...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        average_loss = []
        for epoch in range(NUM_EPOCHS_W2V):
            with open(dataset_path, encoding='utf-8') as dataset:
                for batch_inputs, batch_labels in word2vec_batch_generator(dataset,
                                                                           ngram_size,
                                                                           BATCH_SIZE_W2V,
                                                                           WINDOW_SIZE_W2V,
                                                                           word_to_idx):
                    _, loss_val = sess.run(
                        [optimizer, loss],
                        feed_dict={inputs: batch_inputs,
                                   labels: batch_labels}
                    )

                    average_loss.append(loss_val)

                time = ""
                if (epoch + 1) % 10 == 0:
                    time = "[%s]" % (str(datetime.now()))
                print("Epoch %d - average loss: %f %s" % (epoch + 1, np.mean(average_loss), time))
                average_loss = []

        print("Exporting embeddings... [%s]" % (str(datetime.now())))
        embedding_matrix = sess.run(embeddings)
        with open(embeddings_path, 'w', encoding='utf-8') as embeddings_file:
            for i in range(len(word_to_idx)):
                emb = embedding_matrix[i, :]
                word = idx_to_words[i]
                embeddings_file.write('%s\t%s\n' % (word, ' '.join(map(str, emb))))


def read_embeddings(embeddings_path):
    """
    Reads embeddings from a file generated via the train_embeddings function.
    :param embeddings_path: File path containing the embeddings
    :return: (dictionary mapping words to indices, dictionary mapping indices to words, np.ndarray containing the embedding matrix)
    """
    word_to_idx, idx_to_words = {}, {}
    embeddings_matrix = []
    with open(embeddings_path, encoding='utf-8') as embeddings_file:
        for line in embeddings_file:
            line = line.split("\t")

            word = line[0]
            if word not in word_to_idx:
                idx = len(word_to_idx)
                word_to_idx[word] = idx
                idx_to_words[idx] = word

            vector = line[1]
            embeddings_matrix.append(
                np.asarray(
                    vector.split(),
                    dtype=np.float32
                )
            )

    embeddings_matrix = np.stack(embeddings_matrix)

    return word_to_idx, idx_to_words, embeddings_matrix


# --- --- ---


# --- Related to model training ---


def get_label_dictionaries():
    """
    Returns the label encoding dictionaries for the BIES format.
    :return: (dictionary mapping labels to indices, dictionary mapping indices to labels)
    """
    label_to_idx = {'<PAD>': 0,
                    '-': 1,  # special '-' tag for <S> and </S> tokens
                    'B': 2,
                    'I': 3,
                    'E': 4,
                    'S': 5}
    idx_to_label = dict(zip(label_to_idx.values(), label_to_idx.keys()))

    return label_to_idx, idx_to_label


def ngram_features_preprocessing(batch_inputs, batch_size, ngram_features, word_to_idx, perm=None):
    """
    Tensor preprocessing generalization for multiple ngram features at once.
    :param batch_inputs: raw lines of input file (no spaces)
    :param batch_size: self-explanatory
    :param ngram_features: List of ngrams to consider for batch_inputs generation; ex. ['1', '2'] will produce 1-grams and 2-grams; MUST match len(word_to_idx)
    :param word_to_idx: List of dictionaries mapping words to indices, same order of ngram_features; MUST match len(ngram_features)
    :param perm: np.random.permutation to apply; None if no permutation is required (default)
    :return: (list of batches each being a np.ndarray already padded and shuffled, lengths referring to 1-grams sentences as np.array, padding length aligned to the next batch_size multiple)
    """
    final_batch = []
    max_len = 0

    tmp_batch = []
    for i in range(len(ngram_features)):
        ngram = ngram_features[i]
        dictionary = word_to_idx[i]

        for line in batch_inputs:
            # split to 1-grams and add special tokens for start and end of line
            line = ['<S>'] + split_to_words(line, word_size=ngram) + ['</S>']

            # make it tensor-ready
            preproc = [dictionary[w] if w in dictionary else dictionary['<UNK>'] for w in line]

            # every length only has to refer to 1-grams
            if ngram == 1:
                curr_len = len(preproc)

                # padding up to the longest sentence in each batch
                if curr_len > max_len:
                    max_len = curr_len

            tmp_batch.append(preproc)

        # align max_len to the next batch_size multiple (only once)
        if ngram == 1:
            max_len = batch_size * ceil(float(max_len) / batch_size)

        # batch complete, apply padding
        tmp_batch = pad_sequences(tmp_batch, truncating='pre', padding='post', maxlen=max_len)

        # now shuffle it, if a permutation is provided
        if perm is not None:
            tmp_batch = tmp_batch[perm]

        # finally append it to the list of batches to be returned
        final_batch.append(tmp_batch)
        tmp_batch = []

    return final_batch, np.count_nonzero(final_batch[0], axis=-1), max_len


def generate_batches(dataset_input, dataset_label, batch_size, label_to_idx, ngram_features, word_to_idx, to_shuffle=True, testing=False):
    """
    Generate batches reading two files simultaneously. Takes care of all the pre-processing needed.
    :param dataset_input: File path to the input file dataset (no-spaces)
    :param dataset_label: File path to the label file dataset (BIES format)
    :param batch_size: Size of batches to create
    :param label_to_idx: Dictionary mapping labels to indices
    :param ngram_features: List of ngrams to consider for batch_inputs generation; ex. ['1', '2'] will produce 1-grams and 2-grams; MUST match len(word_to_idx)
    :param word_to_idx: List of dictionaries mapping words to indices, same order of ngram_features; MUST match len(ngram_features)
    :param to_shuffle: True if batches have to be shuffled; False otherwise
    :return: (batch_input, batch_label, batch_lengths) in a generator fashion
    """
    assert len(ngram_features) > 0, "At least one ngram feature has to be specified."
    assert 1 in ngram_features, "1-grams cannot be excluded."
    assert len(ngram_features) == len(word_to_idx), "Non-matching ngram features number with dictionaries provided."

    batch_inputs, batch_labels, batch_lengths = [], [], []
    with open(dataset_input, encoding='utf-8') as inputs_file:
        labels_file = open(dataset_label if not testing else dataset_input, encoding='utf-8')

        # lazily iterate over both files at once
        for line_input, line_label in zip(inputs_file, labels_file):
            line_input = line_input.strip()

            if not testing:
                line_label = line_label.strip()

                # split to 1-grams and add special tokens for start and end of line
                line_label = ['-'] + split_to_words(line_label, word_size=1) + ['-']
                preproc_label = [label_to_idx[l] for l in line_label]

            # add to batches
            batch_inputs.append(line_input)

            if not testing:
                batch_labels.append(preproc_label)

            # if batch ready, apply padding and shuffle it (if required)
            if len(batch_inputs) == batch_size:
                perm = np.random.permutation(batch_size) if to_shuffle else None
                batch_inputs, batch_lengths, max_len = ngram_features_preprocessing(batch_inputs,
                                                                                    batch_size,
                                                                                    ngram_features,
                                                                                    word_to_idx,
                                                                                    perm)

                if not testing:
                    batch_labels = pad_sequences(batch_labels, truncating='pre', padding='post', maxlen=max_len)

                if to_shuffle and not testing:
                    batch_labels = batch_labels[perm]

                yield batch_inputs, batch_labels, batch_lengths

                # start new batch
                batch_inputs, batch_labels, batch_lengths = [], [], []

        if testing:
            labels_file.close()

    # same treatment for incomplete batches
    if len(batch_inputs) > 0:
        perm = np.random.permutation(len(batch_inputs)) if to_shuffle else None
        batch_inputs, batch_lengths, max_len = ngram_features_preprocessing(batch_inputs,
                                                                            batch_size,
                                                                            ngram_features,
                                                                            word_to_idx,
                                                                            perm)

        if not testing:
            batch_labels = pad_sequences(batch_labels, truncating='pre', padding='post', maxlen=max_len)

        if to_shuffle and not testing:
            batch_labels = batch_labels[perm]

        yield batch_inputs, batch_labels, batch_lengths


def get_train_dev_test(dataset_list, output_dir, size=-1, partition=0.8, train_test_only=False):
    """
    Generates small train, dev, and test sets to be used for grid search.
    :param dataset_list: List of tuples, each of them containing input file and label file
    :param output_dir: Directory where to store the train, dev, and test files generated
    :param size: Number of lines to be included in train, dev, test sets combined; this parameter is a lower bound of the actual number of lines; if -1 read whole file (default)
    :param partition: Percentage to split train and dev + test sets into
    :param train_test_only: True only outputs train and test files, without dev; False usual behaviour (default)
    :return: List of tuples, each containing input and label file paths; follows this ordering: train, dev, test.
    """
    inputs = []
    labels = []

    each_size = ceil(float(size) / len(dataset_list)) if size != -1 else -1
    for dataset_inputs, dataset_labels in dataset_list:
        num_lines = 0
        with \
                open(dataset_inputs, encoding='utf-8') as inputs_file,\
                open(dataset_labels, encoding='utf-8') as labels_file:
            for line_inputs, line_labels in zip(inputs_file, labels_file):
                if each_size != -1 and num_lines == each_size:
                    break

                inputs.append(line_inputs.strip())
                labels.append(line_labels.strip())

                num_lines += 1

    train_inputs, dev_inputs, train_labels, dev_labels = train_test_split(inputs,
                                                                          labels,
                                                                          train_size=partition)

    if not train_test_only:
        dev_inputs, test_inputs, dev_labels, test_labels = train_test_split(dev_inputs,
                                                                            dev_labels,
                                                                            train_size=(1.0-partition)/2)

    # write to files
    train_file1, train_file2 = output_dir + "/train_inputs.utf8", output_dir + "/train_labels.utf8"
    dev_file1, dev_file2 = output_dir + "/dev_inputs.utf8", output_dir + "/dev_labels.utf8"
    test_file1, test_file2 = output_dir + "/test_inputs.utf8", output_dir + "/test_labels.utf8"

    with \
            open(train_file1, mode='w', encoding='utf-8') as file_inputs, \
            open(train_file2, mode='w', encoding='utf-8') as file_labels:
        file_inputs.writelines("%s\n" % l for l in train_inputs)
        file_labels.writelines("%s\n" % l for l in train_labels)

    with \
            open(dev_file1, mode='w', encoding='utf-8') as file_inputs, \
            open(dev_file2, mode='w', encoding='utf-8') as file_labels:
        file_inputs.writelines("%s\n" % l for l in dev_inputs)
        file_labels.writelines("%s\n" % l for l in dev_labels)

    if not train_test_only:
        with \
                open(test_file1, mode='w', encoding='utf-8') as file_inputs, \
                open(test_file2, mode='w', encoding='utf-8') as file_labels:
            file_inputs.writelines("%s\n" % l for l in test_inputs)
            file_labels.writelines("%s\n" % l for l in test_labels)

    ret = [(train_file1, train_file2), (dev_file1, dev_file2)]
    if not train_test_only:
        ret = ret + [(test_file1, test_file2)]

    return ret

# --- --- ---


if __name__ == '__main__':
    # vocabularies
    #make_vocab(
    #    "../resources/train/merged_input.utf8",
    #    "../resources/train/vocab_1grams.utf8",
    #    150_000,
    #    1
    #)

    #make_vocab(
    #    "../resources/train/merged_input.utf8",
    #    "../resources/train/vocab_2grams.utf8",
    #    350_000,
    #    2
    #)

    #make_vocab(
    #    "../resources/train/merged_input.utf8",
    #    "../resources/train/vocab_3grams.utf8",
    #    500_000,
    #    3
    #)
    # --- --- ---


    # embeddings training
    #tf.reset_default_graph()
    #print("Embeddings - 1-grams [%s]" % (str(datetime.now())))

    #train_embeddings(
    #    "../resources/train/merged_input.utf8",
    #    "../resources/train/vocab_1grams.utf8",
    #    "../resources/train/embeddings_1grams_2.utf8",
    #    64,
    #    1
    #)

    #tf.reset_default_graph()
    #print("Embeddings - 2-grams [%s]" % (str(datetime.now())))

    #train_embeddings(
    #    "../resources/train/merged_input.utf8",
    #    "../resources/train/vocab_2grams.utf8",
    #    "../resources/train/embeddings_2grams_2.utf8",
    #    64,
    #    2
    #)

    #tf.reset_default_graph()
    #print("Embeddings - 3-grams [%s]" % (str(datetime.now())))

    #train_embeddings(
    #    "../resources/train/merged_input.utf8",
    #    "../resources/train/vocab_3grams.utf8",
    #    "../resources/train/embeddings_3grams_2.utf8",
    #    64,
    #    3
    #)

    # --- --- ---

    # read embeddings
    # read_embeddings("../resources/train/embeddings_2grams.utf8")
    # --- --- ---

    # toy train, dev, test sets
    #get_train_dev_test(dataset_list=[("../resources/train/as_training_simpl_input.utf8", "../resources/train/as_training_simpl_label.utf8"),
    #                                 ("../resources/train/cityu_training_simpl_input.utf8", "../resources/train/cityu_training_simpl_label.utf8"),
    #                                 ("../resources/train/msr_training_input.utf8", "../resources/train/msr_training_label.utf8"),
    #                                 ("../resources/train/pku_training_input.utf8", "../resources/train/pku_training_label.utf8")],
    #                   output_dir="../resources/train/",
    #                   size=20_000)
    # --- --- ---

    # real dev sets
    #get_train_dev_test(dataset_list=[("../resources/dev/pku_test_gold_input.utf8", "../resources/dev/pku_test_gold_label.utf8")],
    #                   output_dir="../resources/dev/",
    #                   partition=0.2,
    #                   train_test_only=True)
    # --- --- ---

    pass
