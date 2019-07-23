from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
from datetime import datetime

# custom modules
import utils as u
import model as m


# --- Common utilities ---
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def compute_accuracy(predictions, labels):
    """
    Computes accuracy depending on how many labels are correctly assigned, averaged.
    :param predictions: Labels predicted by the model
    :param labels: Gold labels
    :return: accuracy value, as float
    """
    accuracy = 0.0
    tot_values = 0

    for pred, label in zip(predictions, labels):
        # strip both predictions and labels of special '-' tags and padding
        pred = pred[1:np.count_nonzero(pred)-1]
        label = label[1:np.count_nonzero(label)-1]

        accuracy += np.count_nonzero(pred == label)*1.0
        tot_values += np.count_nonzero(pred)

    return accuracy / tot_values


def add_summary(writer, name, value, global_step):
    """
    Utility function to track the model's progess in TensorBoard
    :param writer: tf.summary.FileWriter instance
    :param name: Value label to be shown in TensorBoard
    :param value: Value to append for the current step
    :param global_step: Current step for which the value has to be considered
    :return: None
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
    writer.add_summary(summary, global_step=global_step)

# --- --- ---


# --- Related to grid search ---

# Hyper-parameter lists for greedy optimization w.r.t. accuracy values
GRID_HIDDEN_SIZEs = [64, 96]
GRID_LRs = [0.001, 0.005]
GRID_DROPOUTs = [0.2, 0.35]
GRID_REC_DROPOUTs = [0.1, 0.2]


def grid_search(resources_path):
    """
    Performs hyper-parameters' grid search and prints outputs both to the standard output and on a file.
    WARNING: may take long time
    :return: None
    """
    global GRID_HIDDEN_SIZEs, GRID_LRs, GRID_DROPOUTs, GRID_REC_DROPOUTs
    accuracies = {}

    time = "[%s]" % (str(datetime.now()))
    print("Grid search started: %s" % time)
    model_ID = 0
    for hidden_size in GRID_HIDDEN_SIZEs:
        for lr in GRID_LRs:
            for dropout in GRID_DROPOUTs:
                for rec_dropout in GRID_REC_DROPOUTs:
                    tf.reset_default_graph()

                    key = "ID: %d (hidden_size:%d, lr:%.5f, dropout:%.2f, rec_dropout: %.2f)" % (model_ID, hidden_size, lr, dropout, rec_dropout)
                    acc = train_baseline_model(train_datasets=[resources_path + "/train/train_inputs.utf8", resources_path + "/train/train_labels.utf8"],
                                               dev_datasets=[resources_path + "/train/dev_inputs.utf8", resources_path + "/train/dev_labels.utf8"],
                                               test_datasets=[resources_path + "/train/test_inputs.utf8", resources_path + "/train/test_labels.utf8"],
                                               model_path=resources_path + "/base_model_%d/base_model.ckpt" % model_ID,
                                               model_ID=model_ID,
                                               epochs=10,
                                               batch_size=256,
                                               hidden_size=hidden_size,
                                               lr=lr,
                                               dropout=dropout,
                                               rec_dropout=rec_dropout)
                    accuracies[key] = acc

                    model_ID += 1

    print("---")
    with open(resources_path + "/train/grid_search.txt", mode='w') as grid_file:
        values = "Model\tTrain acc\tDev acc\tTest acc\n"
        grid_file.write(values)
        for settings, acc in accuracies.items():
            values = "%s\t%.5f\t%.5f\t%.5f\n" % (settings, acc[0], acc[1], acc[2])
            print(values)
            grid_file.write(values)

# --- --- ---

# --- Related to model training ---
# Hyper-parameters - found via grid search
NUM_EPOCHS = 25
BATCH_SIZE = 256
HIDDEN_SIZE = 96
LEARNING_RATE = 0.005
DROPOUT = 0.2
REC_DROPOUT = 0.1
# ---


def train_baseline_model(train_datasets,
                         dev_datasets,
                         test_datasets,
                         model_path,
                         model_ID=0,
                         epochs=NUM_EPOCHS,
                         batch_size=BATCH_SIZE,
                         hidden_size=HIDDEN_SIZE,
                         lr=LEARNING_RATE,
                         dropout=DROPOUT,
                         rec_dropout=REC_DROPOUT):
    """
    Builds and trains a baseline model with the given hyper-parameters and training options.
    :param train_datasets: Tuple (input file, label file) of the training set
    :param dev_datasets: Tuple (input file, label file) of the development set
    :param test_datasets: Tuple (input file, label file) of the test set
    :param model_path: Path to file to be used for saving the trained model
    :param model_ID: self-explanatory
    :param epochs: Number of epochs the model has to train for
    :param batch_size: self-explanatory
    :param hidden_size: Number of hidden units of the model
    :param lr: Learning rate
    :param dropout: Dropout probability for each unit
    :param rec_dropout: Recurrent dropout probability for each unit
    :return: tuple (last recorded accuracy on the training set, last recorded accuracy on dev set, accuracy on test set)
    """

    print("Creating model...")
    x_1grams, x_2grams, y, \
        keep_pr, recurrent_keep_pr, \
        lengths, train, \
        loss, preds = m.get_baseline_model(
            pretrained_emb_1grams=emb_matrix_1grams,
            pretrained_emb_2grams=emb_matrix_2grams,
            hidden_size=hidden_size,
            y_size=len(labels_to_idx),
            learning_rate=lr
        )

    model_params = "ID:%d, units:%d, lr:%.5f, dr:%.2f, rec_dr: %.2f]" % (model_ID, hidden_size, lr, dropout, rec_dropout)
    saver = tf.train.Saver()

    with \
            tf.Session() as sess,\
            tf.summary.FileWriter("logging/base_model %s" % model_params, sess.graph) as train_logger:

        sess.run(tf.global_variables_initializer())

        # keep track of last accuracies recorded to allow greedy optimization
        accuracies = []

        print("Starting training...")
        for epoch in range(epochs):
            time = "[%s]" % (str(datetime.now()))
            print("Epoch %d %s" % (epoch, time))

            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_inputs,\
                batch_labels,\
                batch_lengths in u.generate_batches(dataset_input=train_datasets[0],
                                                    dataset_label=train_datasets[1],
                                                    batch_size=batch_size,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams]):

                _, loss_val, preds_val = sess.run(
                    [train, loss, preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               y: batch_labels,
                               lengths: batch_lengths,
                               keep_pr: 1.0-dropout,
                               recurrent_keep_pr: 1.0-rec_dropout}
                )

                accumulated_loss += loss_val
                accumulated_acc += compute_accuracy(preds_val, batch_labels)
                iterations += 1

            accumulated_loss /= iterations
            accumulated_acc /= iterations

            print("\t- Train set:\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

            # only store the last accuracy recorded
            if epoch == epochs-1:
                accuracies.append(accumulated_acc)

            add_summary(train_logger,
                        "train loss",
                        accumulated_loss,
                        epoch)

            add_summary(train_logger,
                        "train acc",
                        accumulated_acc,
                        epoch)

            # dev set
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_inputs,\
                batch_labels,\
                batch_lengths in u.generate_batches(dataset_input=dev_datasets[0],
                                                    dataset_label=dev_datasets[1],
                                                    batch_size=batch_size,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams]):

                loss_val, preds_val = sess.run(
                    [loss, preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               y: batch_labels,
                               lengths: batch_lengths,
                               keep_pr: 1.0,
                               recurrent_keep_pr: 1.0}
                )

                accumulated_loss += loss_val
                accumulated_acc += compute_accuracy(preds_val, batch_labels)
                iterations += 1

            accumulated_loss /= iterations
            accumulated_acc /= iterations

            print("\t- Dev set:\t\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

            # only store the last accuracy recorded
            if epoch == epochs - 1:
                accuracies.append(accumulated_acc)

            # save model periodically
            if epoch % 10 == 0:
                saver.save(sess, model_path)
                time = "[%s]" % (str(datetime.now()))
                print("Model saved: %s" % time)

            add_summary(train_logger,
                        "dev loss",
                        accumulated_loss,
                        epoch)

            add_summary(train_logger,
                        "dev acc",
                        accumulated_acc,
                        epoch)

        # save model after training ended
        saver.save(sess, model_path)

        time = "[%s]" % (str(datetime.now()))
        print("Training ended %s.\nEvaluating model on test set..." % time)
        accumulated_loss = 0
        accumulated_acc = 0
        iterations = 0

        for batch_inputs, \
            batch_labels, \
            batch_lengths in u.generate_batches(dataset_input=test_datasets[0],
                                                dataset_label=test_datasets[1],
                                                batch_size=batch_size,
                                                label_to_idx=labels_to_idx,
                                                ngram_features=[1, 2],
                                                word_to_idx=[word_to_idx_1grams, word_to_idx_2grams],
                                                to_shuffle=False):
            loss_val, preds_val = sess.run(
                [loss, preds],
                feed_dict={x_1grams: batch_inputs[0],
                           x_2grams: batch_inputs[1],
                           y: batch_labels,
                           lengths: batch_lengths,
                           keep_pr: 1.0,
                           recurrent_keep_pr: 1.0}
            )

            accumulated_loss += loss_val
            accumulated_acc += compute_accuracy(preds_val, batch_labels)
            iterations += 1

        accumulated_loss /= iterations
        accumulated_acc /= iterations

        print("Test set:\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

        add_summary(train_logger,
                    "test loss",
                    accumulated_loss,
                    epoch)

        add_summary(train_logger,
                    "test acc",
                    accumulated_acc,
                    epoch)

        # always store accuracy on test set
        accuracies.append(accumulated_acc)

    return accuracies


def train_3grams_model(train_datasets,
                       dev_datasets,
                       test_datasets,
                       model_path,
                       model_ID=0,
                       epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE,
                       hidden_size=HIDDEN_SIZE,
                       lr=LEARNING_RATE,
                       dropout=DROPOUT,
                       rec_dropout=REC_DROPOUT):
    """
    Builds and trains a 3-grams model with the given hyper-parameters and training options.
    :param train_datasets: Tuple (input file, label file) of the training set
    :param dev_datasets: Tuple (input file, label file) of the development set
    :param test_datasets: Tuple (input file, label file) of the test set
    :param model_path: Path to file to be used for saving the trained model
    :param model_ID: self-explanatory
    :param epochs: Number of epochs the model has to train for
    :param batch_size: self-explanatory
    :param hidden_size: Number of hidden units of the model
    :param lr: Learning rate
    :param dropout: Dropout probability for each unit
    :param rec_dropout: Recurrent dropout probability for each unit
    :return: tuple (last recorded accuracy on the training set, last recorded accuracy on dev set, accuracy on test set)
    """

    print("Creating model...")
    x_1grams, x_2grams, x_3grams, y, \
        keep_pr, recurrent_keep_pr, \
        lengths, train, \
        loss, preds = m.get_3grams_model(
            pretrained_emb_1grams=emb_matrix_1grams,
            pretrained_emb_2grams=emb_matrix_2grams,
            pretrained_emb_3grams=emb_matrix_3grams,
            hidden_size=hidden_size,
            y_size=len(labels_to_idx),
            learning_rate=lr
        )

    model_params = "ID:%d, units:%d, lr:%.5f, dr:%.2f, rec_dr: %.2f]" % (model_ID, hidden_size, lr, dropout, rec_dropout)
    saver = tf.train.Saver()

    with \
            tf.Session() as sess,\
            tf.summary.FileWriter("logging/3grams_model %s" % model_params, sess.graph) as train_logger:

        sess.run(tf.global_variables_initializer())

        # keep track of last accuracies recorded to allow greedy optimization
        accuracies = []

        print("Starting training...")
        for epoch in range(epochs):
            time = "[%s]" % (str(datetime.now()))
            print("Epoch %d %s" % (epoch, time))

            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_inputs,\
                batch_labels,\
                batch_lengths in u.generate_batches(dataset_input=train_datasets[0],
                                                    dataset_label=train_datasets[1],
                                                    batch_size=batch_size,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2, 3],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams, word_to_idx_3grams]):

                _, loss_val, preds_val = sess.run(
                    [train, loss, preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               x_3grams: batch_inputs[2],
                               y: batch_labels,
                               lengths: batch_lengths,
                               keep_pr: 1.0-dropout,
                               recurrent_keep_pr: 1.0-rec_dropout}
                )

                accumulated_loss += loss_val
                accumulated_acc += compute_accuracy(preds_val, batch_labels)
                iterations += 1

            accumulated_loss /= iterations
            accumulated_acc /= iterations

            print("\t- Train set:\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

            # only store the last accuracy recorded
            if epoch == epochs-1:
                accuracies.append(accumulated_acc)

            add_summary(train_logger,
                        "train loss",
                        accumulated_loss,
                        epoch)

            add_summary(train_logger,
                        "train acc",
                        accumulated_acc,
                        epoch)

            # dev set
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_inputs,\
                batch_labels,\
                batch_lengths in u.generate_batches(dataset_input=dev_datasets[0],
                                                    dataset_label=dev_datasets[1],
                                                    batch_size=batch_size,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2, 3],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams, word_to_idx_3grams]):

                loss_val, preds_val = sess.run(
                    [loss, preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               x_3grams: batch_inputs[2],
                               y: batch_labels,
                               lengths: batch_lengths,
                               keep_pr: 1.0,
                               recurrent_keep_pr: 1.0}
                )

                accumulated_loss += loss_val
                accumulated_acc += compute_accuracy(preds_val, batch_labels)
                iterations += 1

            accumulated_loss /= iterations
            accumulated_acc /= iterations

            print("\t- Dev set:\t\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

            # only store the last accuracy recorded
            if epoch == epochs - 1:
                accuracies.append(accumulated_acc)

            # save model periodically
            if epoch % 10 == 0:
                saver.save(sess, model_path)
                time = "[%s]" % (str(datetime.now()))
                print("Model saved: %s" % time)

            add_summary(train_logger,
                        "dev loss",
                        accumulated_loss,
                        epoch)

            add_summary(train_logger,
                        "dev acc",
                        accumulated_acc,
                        epoch)

        # save model after training ended
        saver.save(sess, model_path)

        time = "[%s]" % (str(datetime.now()))
        print("Training ended %s.\nEvaluating model on test set..." % time)
        accumulated_loss = 0
        accumulated_acc = 0
        iterations = 0

        for batch_inputs, \
            batch_labels, \
            batch_lengths in u.generate_batches(dataset_input=test_datasets[0],
                                                dataset_label=test_datasets[1],
                                                batch_size=batch_size,
                                                label_to_idx=labels_to_idx,
                                                ngram_features=[1, 2, 3],
                                                word_to_idx=[word_to_idx_1grams, word_to_idx_2grams, word_to_idx_3grams],
                                                to_shuffle=False):
            loss_val, preds_val = sess.run(
                [loss, preds],
                feed_dict={x_1grams: batch_inputs[0],
                           x_2grams: batch_inputs[1],
                           x_3grams: batch_inputs[2],
                           y: batch_labels,
                           lengths: batch_lengths,
                           keep_pr: 1.0,
                           recurrent_keep_pr: 1.0}
            )

            accumulated_loss += loss_val
            accumulated_acc += compute_accuracy(preds_val, batch_labels)
            iterations += 1

        accumulated_loss /= iterations
        accumulated_acc /= iterations

        print("Test set:\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

        add_summary(train_logger,
                    "test loss",
                    accumulated_loss,
                    epoch)

        add_summary(train_logger,
                    "test acc",
                    accumulated_acc,
                    epoch)

        # always store accuracy on test set
        accuracies.append(accumulated_acc)

    return accuracies


def train_layered_model(train_datasets,
                        dev_datasets,
                        test_datasets,
                        model_path,
                        model_ID=0,
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        hidden_size=HIDDEN_SIZE,
                        layers=1,
                        lr=LEARNING_RATE,
                        dropout=DROPOUT,
                        rec_dropout=REC_DROPOUT):
    """
    Builds and trains a 2 layers model with the given hyper-parameters and training options.
    :param train_datasets: Tuple (input file, label file) of the training set
    :param dev_datasets: Tuple (input file, label file) of the development set
    :param test_datasets: Tuple (input file, label file) of the test set
    :param model_path: Path to file to be used for saving the trained model
    :param model_ID: self-explanatory
    :param epochs: Number of epochs the model has to train for
    :param batch_size: self-explanatory
    :param hidden_size: Number of hidden units of the model
    :param lr: Learning rate
    :param dropout: Dropout probability for each unit
    :param rec_dropout: Recurrent dropout probability for each unit
    :return: tuple (last recorded accuracy on the training set, last recorded accuracy on dev set, accuracy on test set)
    """

    print("Creating model...")
    x_1grams, x_2grams, y, \
        keep_pr, recurrent_keep_pr, \
        lengths, train, \
        loss, preds = m.get_layered_model(
            pretrained_emb_1grams=emb_matrix_1grams,
            pretrained_emb_2grams=emb_matrix_2grams,
            hidden_size=hidden_size,
            layers=layers,
            y_size=len(labels_to_idx),
            learning_rate=lr
        )

    model_params = "ID:%d, units:%d, lr:%.5f, dr:%.2f, rec_dr: %.2f]" % (model_ID, hidden_size, lr, dropout, rec_dropout)
    saver = tf.train.Saver()

    with \
            tf.Session() as sess,\
            tf.summary.FileWriter("logging/%dlayers_model %s" % ((layers+1), model_params), sess.graph) as train_logger:

        sess.run(tf.global_variables_initializer())

        # keep track of last accuracies recorded to allow greedy optimization
        accuracies = []

        print("Starting training...")
        for epoch in range(epochs):
            time = "[%s]" % (str(datetime.now()))
            print("Epoch %d %s" % (epoch, time))

            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_inputs,\
                batch_labels,\
                batch_lengths in u.generate_batches(dataset_input=train_datasets[0],
                                                    dataset_label=train_datasets[1],
                                                    batch_size=batch_size,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams]):

                _, loss_val, preds_val = sess.run(
                    [train, loss, preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               y: batch_labels,
                               lengths: batch_lengths,
                               keep_pr: 1.0-dropout,
                               recurrent_keep_pr: 1.0-rec_dropout}
                )

                accumulated_loss += loss_val
                accumulated_acc += compute_accuracy(preds_val, batch_labels)
                iterations += 1

            accumulated_loss /= iterations
            accumulated_acc /= iterations

            print("\t- Train set:\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

            # only store the last accuracy recorded
            if epoch == epochs-1:
                accuracies.append(accumulated_acc)

            add_summary(train_logger,
                        "train loss",
                        accumulated_loss,
                        epoch)

            add_summary(train_logger,
                        "train acc",
                        accumulated_acc,
                        epoch)

            # dev set
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_inputs,\
                batch_labels,\
                batch_lengths in u.generate_batches(dataset_input=dev_datasets[0],
                                                    dataset_label=dev_datasets[1],
                                                    batch_size=batch_size,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams]):

                loss_val, preds_val = sess.run(
                    [loss, preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               y: batch_labels,
                               lengths: batch_lengths,
                               keep_pr: 1.0,
                               recurrent_keep_pr: 1.0}
                )

                accumulated_loss += loss_val
                accumulated_acc += compute_accuracy(preds_val, batch_labels)
                iterations += 1

            accumulated_loss /= iterations
            accumulated_acc /= iterations

            print("\t- Dev set:\t\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

            # only store the last accuracy recorded
            if epoch == epochs - 1:
                accuracies.append(accumulated_acc)

            # save model periodically
            if epoch % 10 == 0:
                saver.save(sess, model_path)
                time = "[%s]" % (str(datetime.now()))
                print("Model saved: %s" % time)

            add_summary(train_logger,
                        "dev loss",
                        accumulated_loss,
                        epoch)

            add_summary(train_logger,
                        "dev acc",
                        accumulated_acc,
                        epoch)

        # save model after training ended
        saver.save(sess, model_path)

        time = "[%s]" % (str(datetime.now()))
        print("Training ended %s.\nEvaluating model on test set..." % time)
        accumulated_loss = 0
        accumulated_acc = 0
        iterations = 0

        for batch_inputs, \
            batch_labels, \
            batch_lengths in u.generate_batches(dataset_input=test_datasets[0],
                                                dataset_label=test_datasets[1],
                                                batch_size=batch_size,
                                                label_to_idx=labels_to_idx,
                                                ngram_features=[1, 2],
                                                word_to_idx=[word_to_idx_1grams, word_to_idx_2grams],
                                                to_shuffle=False):
            loss_val, preds_val = sess.run(
                [loss, preds],
                feed_dict={x_1grams: batch_inputs[0],
                           x_2grams: batch_inputs[1],
                           y: batch_labels,
                           lengths: batch_lengths,
                           keep_pr: 1.0,
                           recurrent_keep_pr: 1.0}
            )

            accumulated_loss += loss_val
            accumulated_acc += compute_accuracy(preds_val, batch_labels)
            iterations += 1

        accumulated_loss /= iterations
        accumulated_acc /= iterations

        print("Test set:\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

        add_summary(train_logger,
                    "test loss",
                    accumulated_loss,
                    epoch)

        add_summary(train_logger,
                    "test acc",
                    accumulated_acc,
                    epoch)

        # always store accuracy on test set
        accuracies.append(accumulated_acc)

    return accuracies


def train_3grams_layered_model(train_datasets,
                               dev_datasets,
                               test_datasets,
                               model_path,
                               model_ID=0,
                               epochs=NUM_EPOCHS,
                               batch_size=BATCH_SIZE,
                               hidden_size=HIDDEN_SIZE,
                               lr=LEARNING_RATE,
                               dropout=DROPOUT,
                               rec_dropout=REC_DROPOUT):
    """
    Builds and trains a 2 layers 3-grams model with the given hyper-parameters and training options.
    :param train_datasets: Tuple (input file, label file) of the training set
    :param dev_datasets: Tuple (input file, label file) of the development set
    :param test_datasets: Tuple (input file, label file) of the test set
    :param model_path: Path to file to be used for saving the trained model
    :param model_ID: self-explanatory
    :param epochs: Number of epochs the model has to train for
    :param batch_size: self-explanatory
    :param hidden_size: Number of hidden units of the model
    :param lr: Learning rate
    :param dropout: Dropout probability for each unit
    :param rec_dropout: Recurrent dropout probability for each unit
    :return: tuple (last recorded accuracy on the training set, last recorded accuracy on dev set, accuracy on test set)
    """

    print("Creating model...")
    x_1grams, x_2grams, x_3grams, y, \
        keep_pr, recurrent_keep_pr, \
        lengths, train, \
        loss, preds = m.get_3grams_layered_model(
            pretrained_emb_1grams=emb_matrix_1grams,
            pretrained_emb_2grams=emb_matrix_2grams,
            pretrained_emb_3grams=emb_matrix_3grams,
            hidden_size=hidden_size,
            layers=1,
            y_size=len(labels_to_idx),
            learning_rate=lr
        )

    model_params = "ID:%d, units:%d, lr:%.5f, dr:%.2f, rec_dr: %.2f]" % (model_ID, hidden_size, lr, dropout, rec_dropout)
    saver = tf.train.Saver()

    with \
            tf.Session() as sess,\
            tf.summary.FileWriter("logging/3grams_2layers_model %s" % model_params, sess.graph) as train_logger:

        sess.run(tf.global_variables_initializer())

        # keep track of last accuracies recorded to allow greedy optimization
        accuracies = []

        print("Starting training...")
        for epoch in range(epochs):
            time = "[%s]" % (str(datetime.now()))
            print("Epoch %d %s" % (epoch, time))

            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_inputs,\
                batch_labels,\
                batch_lengths in u.generate_batches(dataset_input=train_datasets[0],
                                                    dataset_label=train_datasets[1],
                                                    batch_size=batch_size,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2, 3],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams, word_to_idx_3grams]):

                _, loss_val, preds_val = sess.run(
                    [train, loss, preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               x_3grams: batch_inputs[2],
                               y: batch_labels,
                               lengths: batch_lengths,
                               keep_pr: 1.0-dropout,
                               recurrent_keep_pr: 1.0-rec_dropout}
                )

                accumulated_loss += loss_val
                accumulated_acc += compute_accuracy(preds_val, batch_labels)
                iterations += 1

            accumulated_loss /= iterations
            accumulated_acc /= iterations

            print("\t- Train set:\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

            # only store the last accuracy recorded
            if epoch == epochs-1:
                accuracies.append(accumulated_acc)

            add_summary(train_logger,
                        "train loss",
                        accumulated_loss,
                        epoch)

            add_summary(train_logger,
                        "train acc",
                        accumulated_acc,
                        epoch)

            # dev set
            accumulated_loss = 0
            accumulated_acc = 0
            iterations = 0

            for batch_inputs,\
                batch_labels,\
                batch_lengths in u.generate_batches(dataset_input=dev_datasets[0],
                                                    dataset_label=dev_datasets[1],
                                                    batch_size=batch_size,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2, 3],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams, word_to_idx_3grams]):

                loss_val, preds_val = sess.run(
                    [loss, preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               x_3grams: batch_inputs[2],
                               y: batch_labels,
                               lengths: batch_lengths,
                               keep_pr: 1.0,
                               recurrent_keep_pr: 1.0}
                )

                accumulated_loss += loss_val
                accumulated_acc += compute_accuracy(preds_val, batch_labels)
                iterations += 1

            accumulated_loss /= iterations
            accumulated_acc /= iterations

            print("\t- Dev set:\t\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

            # only store the last accuracy recorded
            if epoch == epochs - 1:
                accuracies.append(accumulated_acc)

            # save model periodically
            if epoch % 10 == 0:
                saver.save(sess, model_path)
                time = "[%s]" % (str(datetime.now()))
                print("Model saved: %s" % time)

            add_summary(train_logger,
                        "dev loss",
                        accumulated_loss,
                        epoch)

            add_summary(train_logger,
                        "dev acc",
                        accumulated_acc,
                        epoch)

        # save model after training ended
        saver.save(sess, model_path)

        time = "[%s]" % (str(datetime.now()))
        print("Training ended %s.\nEvaluating model on test set..." % time)
        accumulated_loss = 0
        accumulated_acc = 0
        iterations = 0

        for batch_inputs, \
            batch_labels, \
            batch_lengths in u.generate_batches(dataset_input=test_datasets[0],
                                                dataset_label=test_datasets[1],
                                                batch_size=batch_size,
                                                label_to_idx=labels_to_idx,
                                                ngram_features=[1, 2, 3],
                                                word_to_idx=[word_to_idx_1grams, word_to_idx_2grams, word_to_idx_3grams],
                                                to_shuffle=False):
            loss_val, preds_val = sess.run(
                [loss, preds],
                feed_dict={x_1grams: batch_inputs[0],
                           x_2grams: batch_inputs[1],
                           x_3grams: batch_inputs[2],
                           y: batch_labels,
                           lengths: batch_lengths,
                           keep_pr: 1.0,
                           recurrent_keep_pr: 1.0}
            )

            accumulated_loss += loss_val
            accumulated_acc += compute_accuracy(preds_val, batch_labels)
            iterations += 1

        accumulated_loss /= iterations
        accumulated_acc /= iterations

        print("Test set:\tLoss: %.5f\tAcc: %.5f" % (accumulated_loss, accumulated_acc))

        add_summary(train_logger,
                    "test loss",
                    accumulated_loss,
                    epoch)

        add_summary(train_logger,
                    "test acc",
                    accumulated_acc,
                    epoch)

        # always store accuracy on test set
        accuracies.append(accumulated_acc)

    return accuracies

# --- --- ---


if __name__ == '__main__':
    """
    Trains baseline and 2 Bi-LSTM layer models singularly for each dataset.
    """
    resources_path = parse_args().resources_path
    emb_path_1grams = resources_path + "/train/embeddings_1grams.utf8"
    emb_path_2grams = resources_path + "/train/embeddings_2grams.utf8"
    emb_path_3grams = resources_path + "/train/embeddings_3grams.utf8"

    word_to_idx_1grams, idx_to_word_1grams, emb_matrix_1grams = u.read_embeddings(emb_path_1grams)
    word_to_idx_2grams, idx_to_word_2grams, emb_matrix_2grams = u.read_embeddings(emb_path_2grams)
    word_to_idx_3grams, idx_to_word_3grams, emb_matrix_3grams = u.read_embeddings(emb_path_3grams)

    labels_to_idx, idx_to_labels = u.get_label_dictionaries()

    #grid_search(resources_path)

    # Train on AS dataset
    tf.reset_default_graph()

    train_baseline_model(train_datasets=[resources_path + "/train/as_training_simpl_input.utf8", resources_path + "/train/as_training_simpl_label.utf8"],
                         dev_datasets=["../resources/dev/as_dev_inputs.utf8", resources_path + "/dev/as_dev_labels.utf8"],
                         test_datasets=[resources_path + "/dev/as_test_inputs.utf8", resources_path + "/dev/as_test_labels.utf8"],
                         model_path=resources_path + "/base_model_as/base_model.ckpt",
                         model_ID=0)

    tf.reset_default_graph()

    train_layered_model(train_datasets=[resources_path + "/train/as_training_simpl_input.utf8", resources_path + "/train/as_training_simpl_label.utf8"],
                        dev_datasets=[resources_path + "/dev/as_dev_inputs.utf8", resources_path + "/dev/as_dev_labels.utf8"],
                        test_datasets=[resources_path + "/dev/as_test_inputs.utf8", resources_path + "/dev/as_test_labels.utf8"],
                        model_path=resources_path + "/2layers_model_as/base_model.ckpt",
                        model_ID=2)

    # Train on CITYU dataset
    tf.reset_default_graph()

    train_baseline_model(train_datasets=[resources_path + "/train/cityu_training_simpl_input.utf8", resources_path + "/train/cityu_training_simpl_label.utf8"],
                         dev_datasets=[resources_path + "/dev/cityu_dev_inputs.utf8", resources_path + "/dev/cityu_dev_labels.utf8"],
                         test_datasets=[resources_path + "/dev/cityu_test_inputs.utf8", resources_path + "/dev/cityu_test_labels.utf8"],
                         model_path=resources_path + "/base_model_cityu/base_model.ckpt",
                         model_ID=10)

    tf.reset_default_graph()

    train_layered_model(train_datasets=[resources_path + "/train/cityu_training_simpl_input.utf8", resources_path + "/train/cityu_training_simpl_label.utf8"],
                        dev_datasets=[resources_path + "/dev/cityu_dev_inputs.utf8", resources_path + "/dev/cityu_dev_labels.utf8"],
                        test_datasets=[resources_path + "/dev/cityu_test_inputs.utf8", resources_path + "/dev/cityu_test_labels.utf8"],
                        model_path=resources_path + "/2layers_model_cityu/base_model.ckpt",
                        model_ID=12)

    # Train on MSR dataset
    tf.reset_default_graph()

    train_baseline_model(train_datasets=[resources_path + "/train/msr_training_input.utf8", resources_path + "/train/msr_training_label.utf8"],
                         dev_datasets=[resources_path + "/dev/msr_dev_inputs.utf8", resources_path + "/dev/msr_dev_labels.utf8"],
                         test_datasets=[resources_path + "/dev/msr_test_inputs.utf8", resources_path + "/dev/msr_test_labels.utf8"],
                         model_path=resources_path + "/base_model_msr/base_model.ckpt",
                         model_ID=20)

    tf.reset_default_graph()

    train_layered_model(
        train_datasets=[resources_path + "/train/msr_training_input.utf8", resources_path + "/train/msr_training_label.utf8"],
        dev_datasets=[resources_path + "/dev/msr_dev_inputs.utf8", resources_path + "/dev/msr_dev_labels.utf8"],
        test_datasets=[resources_path + "/dev/msr_test_inputs.utf8", resources_path + "/dev/msr_test_labels.utf8"],
        model_path=resources_path + "/2layers_model_msr/base_model.ckpt",
        model_ID=22)

    # Train on PKU dataset
    tf.reset_default_graph()

    train_baseline_model(train_datasets=[resources_path + "/train/pku_training_input.utf8", resources_path + "/train/pku_training_label.utf8"],
                         dev_datasets=[resources_path + "/dev/pku_dev_inputs.utf8", resources_path + "/dev/pku_dev_labels.utf8"],
                         test_datasets=[resources_path + "/dev/pku_test_inputs.utf8", resources_path + "/dev/pku_test_labels.utf8"],
                         model_path=resources_path + "/base_model_pku/base_model.ckpt",
                         model_ID=30)

    tf.reset_default_graph()

    train_layered_model(
        train_datasets=[resources_path + "/train/pku_training_input.utf8", resources_path + "/train/pku_training_label.utf8"],
        dev_datasets=[resources_path + "/dev/pku_dev_inputs.utf8", resources_path + "/dev/pku_dev_labels.utf8"],
        test_datasets=[resources_path + "/dev/pku_test_inputs.utf8", resources_path + "/dev/pku_test_labels.utf8"],
        model_path=resources_path + "/2layers_model_pku/base_model.ckpt",
        model_ID=32,
        batch_size=128)
