from argparse import ArgumentParser
import tensorflow as tf
import numpy as np

# custom modules
import utils as u
import model as m


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("input_path", help="The path of the input file")
    parser.add_argument("output_path", help="The path of the output file")
    parser.add_argument("resources_path", help="The path of the resources needed to load your model")

    return parser.parse_args()


def predict(input_path, output_path, resources_path):
    """
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the BIES format.
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.

    :param input_path: the path of the input file to predict.
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """

    print("Loading embeddings...")
    emb_path_1grams = resources_path + "/train/embeddings_1grams.utf8"
    emb_path_2grams = resources_path + "/train/embeddings_2grams.utf8"

    word_to_idx_1grams, idx_to_word_1grams, emb_matrix_1grams = u.read_embeddings(emb_path_1grams)
    word_to_idx_2grams, idx_to_word_2grams, emb_matrix_2grams = u.read_embeddings(emb_path_2grams)

    labels_to_idx, idx_to_labels = u.get_label_dictionaries()

    print("Done.")

    tf.reset_default_graph()

    x_1grams, x_2grams, y, \
        keep_pr, recurrent_keep_pr, \
        lengths, train, \
        loss, preds = m.get_layered_model(pretrained_emb_1grams=emb_matrix_1grams,
                                          pretrained_emb_2grams=emb_matrix_2grams,
                                          hidden_size=96,
                                          layers=1,
                                          y_size=len(labels_to_idx),
                                          learning_rate=0.005)

    model_path = resources_path + "/2layers_model_cityu/base_model.ckpt"
    print("Loading model saved in path: %s" % model_path)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)

        with open(output_path, mode='w', encoding='utf-8') as preds_file:
            pass

        print("\nGenerating predictions...")
        predictions = []
        with open(output_path, mode='a', encoding='utf-8') as preds_file:
            for batch_inputs, \
                batch_labels, \
                batch_lengths in u.generate_batches(dataset_input=input_path,
                                                    dataset_label="",
                                                    batch_size=32,
                                                    label_to_idx=labels_to_idx,
                                                    ngram_features=[1, 2],
                                                    word_to_idx=[word_to_idx_1grams, word_to_idx_2grams],
                                                    to_shuffle=False,
                                                    testing=True):
                preds_val = sess.run(
                    [preds],
                    feed_dict={x_1grams: batch_inputs[0],
                               x_2grams: batch_inputs[1],
                               lengths: batch_lengths,
                               keep_pr: 1.0,
                               recurrent_keep_pr: 1.0}
                )

                for p in preds_val[0]:
                    p = p[1:np.count_nonzero(p)-1]
                    p = p.tolist()

                    # default to "S" if some special tag (either '-' or '<PAD>') is predicted
                    p = [idx_to_labels[c] if c > 1 else idx_to_labels[5] for c in p]
                    predictions.append(p)

                if len(predictions) == 128:
                    preds_file.writelines("%s\n" % ''.join(p) for p in predictions)
                    predictions = []

            if len(predictions) > 0:
                preds_file.writelines("%s\n" % ''.join(p) for p in predictions)

    print("Done.\nYour predictions have been stored in path: %s" % output_path)


if __name__ == '__main__':
    args = parse_args()
    predict(args.input_path, args.output_path, args.resources_path)
