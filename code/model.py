import tensorflow as tf


def get_baseline_model(
        pretrained_emb_1grams,
        pretrained_emb_2grams,
        hidden_size,
        y_size,
        learning_rate
):
    """
    Simple BiLSTM model based on 1-grams and 2-grams features, replicating Ma, Ganchev, Weiss's 'State-of-the-art Chinese Word Segmentation with Bi-LSTMs'.
    In particular, this model is based on the non-stacking one (Figure 1, variant a).
    :param pretrained_emb_1grams: Embedding matrix as np.ndarray (for 1-grams)
    :param pretrained_emb_2grams: Embedding matrix as np.ndarray (for 2-grams)
    :param hidden_size: Output size of LSTM networks
    :param y_size: Size of the labels dictionary
    :param learning_rate: self-explanatory
    :return: Tensor operations for interaction with the built graph
    """
    x_1grams = tf.placeholder(tf.int32, shape=[None, None])         # shape: batch_size x max_length (for 1-grams)
    x_2grams = tf.placeholder(tf.int32, shape=[None, None])         # shape: batch_size x max_length (for 2-grams)
    y = tf.placeholder(tf.int32, shape=[None, None])                # shape: batch_size x max_length

    keep_pr = tf.placeholder(tf.float32, shape=[])
    recurrent_keep_pr = tf.placeholder(tf.float32, shape=[])

    # max_length for each sequence
    lengths = tf.placeholder(tf.int32, shape=[None])       # shape batch_size x 1

    # each sentence is wrapped by <S> and </S> tags
    lengths = lengths + 2

    with tf.variable_scope("embeddings"):
        # memory-efficient pre-trained embeddings loading
        embedding_matrix_1grams = tf.get_variable(                                  # shape: vocab_size x embedding_size (same size of the given embedding matrix)
            name="embeddings_1grams",
            shape=[pretrained_emb_1grams.shape[0], pretrained_emb_1grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_1grams),             # initializing to pre-trained_embeddings
            trainable=False                                                         # and making them a constant
        )

        embeddings_1grams = tf.nn.embedding_lookup(embedding_matrix_1grams, x_1grams)       # shape: batch_size x max_length x embedding_size

        # same treatment for 2-grams
        embedding_matrix_2grams = tf.get_variable(
            name="embeddings_2grams",
            shape=[pretrained_emb_2grams.shape[0], pretrained_emb_2grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_2grams),
            trainable=False
        )

        embeddings_2grams = tf.nn.embedding_lookup(embedding_matrix_2grams, x_2grams)

        # concatenate embeddings from 1-grams and 2-grams
        embeddings = tf.concat([embeddings_2grams, embeddings_1grams], axis=-1)             # shape: batch_size x max_length x embedding_size1 + embedding_size2

    with tf.variable_scope("bi-lstm"):
        # forward cell
        ltr_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)         # output shape: batch_size x max_length x hidden_size
        ltr_cell = tf.nn.rnn_cell.DropoutWrapper(
            ltr_cell,
            input_keep_prob=keep_pr,
            output_keep_prob=keep_pr,
            state_keep_prob=recurrent_keep_pr
        )

        # backward cell
        rtl_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)         # output shape: batch_size x max_length x hidden_size
        rtl_cell = tf.nn.rnn_cell.DropoutWrapper(
            rtl_cell,
            input_keep_prob=keep_pr,
            output_keep_prob=keep_pr,
            state_keep_prob=recurrent_keep_pr
        )

        # actual bi-lstm
        (ltr_outputs, rtl_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
            ltr_cell,
            rtl_cell,
            embeddings,
            sequence_length=lengths,
            dtype=tf.float32
        )

        # concat ltr_outputs and rtl_outputs to obtain the overall representation
        lstm_outputs = tf.concat([ltr_outputs, rtl_outputs], axis=-1)   # shape: batch_size x max_length x 2 * hidden_size

    with tf.variable_scope("dense"):
        W = tf.get_variable(                # shape: 2 * hidden_size x labels_size
            name="W",
            shape=[2 * hidden_size, y_size],
            dtype=tf.float32
        )

        b = tf.get_variable(                # shape: labels_size x 1
            name="b",
            shape=[y_size],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )

        max_lengths = tf.shape(lstm_outputs)[1]                                  # save the max_length of each sentence for later

        lstm_outputs_flat = tf.reshape(lstm_outputs, [-1, 2 * hidden_size])      # flattening to shape batch_size x 2 * hidden_size
        logits = lstm_outputs_flat @ W + b                                       # shape: batch_size x labels_size
        logits_batch = tf.reshape(logits, [-1, max_lengths, y_size])             # restoring to shape batch_size x max_length x labels_size

    with tf.variable_scope("loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_batch, labels=y)

        # exclude padding tokens from the gradient optimization
        mask = tf.sequence_mask(lengths)
        losses = tf.boolean_mask(losses, mask)

        loss = tf.reduce_mean(losses)

    with tf.variable_scope("softmax"):
        # return the label with the max score for each token
        preds = tf.cast(tf.argmax(logits_batch, axis=-1), tf.int32)

    with tf.variable_scope("optimizer"):
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return \
        x_1grams, x_2grams, y,\
        keep_pr, recurrent_keep_pr,\
        lengths, train,\
        loss, preds


def get_3grams_model(
        pretrained_emb_1grams,
        pretrained_emb_2grams,
        pretrained_emb_3grams,
        hidden_size,
        y_size,
        learning_rate
):
    """
    Simple BiLSTM model based on 1-grams, 2-grams, and 3-grams features.
    This model only differs from the baseline one from the 3-grams features considered.
    :param pretrained_emb_1grams: Embedding matrix as np.ndarray (for 1-grams)
    :param pretrained_emb_2grams: Embedding matrix as np.ndarray (for 2-grams)
    :param pretrained_emb_3grams: Embedding matrix as np.ndarray (for 3-grams)
    :param hidden_size: Output size of LSTM networks
    :param y_size: Size of the labels dictionary
    :param learning_rate: self-explanatory
    :return: Tensor operations for interaction with the built graph
    """
    x_1grams = tf.placeholder(tf.int32, shape=[None, None])         # shape: batch_size x max_length (for 1-grams)
    x_2grams = tf.placeholder(tf.int32, shape=[None, None])         # shape: batch_size x max_length (for 2-grams)
    x_3grams = tf.placeholder(tf.int32, shape=[None, None])         # shape: batch_size x max_length (for 3-grams)
    y = tf.placeholder(tf.int32, shape=[None, None])                # shape: batch_size x max_length

    keep_pr = tf.placeholder(tf.float32, shape=[])
    recurrent_keep_pr = tf.placeholder(tf.float32, shape=[])

    # max_length for each sequence
    lengths = tf.placeholder(tf.int32, shape=[None])       # shape batch_size x 1

    # each sentence is wrapped by <S> and </S> tags
    lengths = lengths + 2

    with tf.variable_scope("embeddings"):
        # memory-efficient pre-trained embeddings loading
        embedding_matrix_1grams = tf.get_variable(                                  # shape: vocab_size x embedding_size (same size of the given embedding matrix)
            name="embeddings_1grams",
            shape=[pretrained_emb_1grams.shape[0], pretrained_emb_1grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_1grams),             # initializing to pre-trained_embeddings
            trainable=False                                                         # and making them a constant
        )

        embeddings_1grams = tf.nn.embedding_lookup(embedding_matrix_1grams, x_1grams)       # shape: batch_size x max_length x embedding_size

        # same treatment for 2-grams
        embedding_matrix_2grams = tf.get_variable(
            name="embeddings_2grams",
            shape=[pretrained_emb_2grams.shape[0], pretrained_emb_2grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_2grams),
            trainable=False
        )

        embeddings_2grams = tf.nn.embedding_lookup(embedding_matrix_2grams, x_2grams)

        # same treatment for 3-grams
        embedding_matrix_3grams = tf.get_variable(
            name="embeddings_3grams",
            shape=[pretrained_emb_3grams.shape[0], pretrained_emb_3grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_3grams),
            trainable=False
        )

        embeddings_3grams = tf.nn.embedding_lookup(embedding_matrix_3grams, x_3grams)

        # concatenate embeddings from 1-grams and 2-grams
        embeddings = tf.concat([embeddings_3grams, embeddings_2grams, embeddings_1grams], axis=-1)             # shape: batch_size x max_length x embedding_size1 + embedding_size2 + embedding_size3

    with tf.variable_scope("bi-lstm"):
        # forward cell
        ltr_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)         # output shape: batch_size x max_length x hidden_size
        ltr_cell = tf.nn.rnn_cell.DropoutWrapper(
            ltr_cell,
            input_keep_prob=keep_pr,
            output_keep_prob=keep_pr,
            state_keep_prob=recurrent_keep_pr
        )

        # backward cell
        rtl_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)         # output shape: batch_size x max_length x hidden_size
        rtl_cell = tf.nn.rnn_cell.DropoutWrapper(
            rtl_cell,
            input_keep_prob=keep_pr,
            output_keep_prob=keep_pr,
            state_keep_prob=recurrent_keep_pr
        )

        # actual bi-lstm
        (ltr_outputs, rtl_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
            ltr_cell,
            rtl_cell,
            embeddings,
            sequence_length=lengths,
            dtype=tf.float32
        )

        # concat ltr_outputs and rtl_outputs to obtain the overall representation
        lstm_outputs = tf.concat([ltr_outputs, rtl_outputs], axis=-1)   # shape: batch_size x max_length x 2 * hidden_size

    with tf.variable_scope("dense"):
        W = tf.get_variable(                # shape: 2 * hidden_size x labels_size
            name="W",
            shape=[2 * hidden_size, y_size],
            dtype=tf.float32
        )

        b = tf.get_variable(                # shape: labels_size x 1
            name="b",
            shape=[y_size],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )

        max_lengths = tf.shape(lstm_outputs)[1]                                  # save the max_length of each sentence for later

        lstm_outputs_flat = tf.reshape(lstm_outputs, [-1, 2 * hidden_size])      # flattening to shape batch_size x 2 * hidden_size
        logits = lstm_outputs_flat @ W + b                                       # shape: batch_size x labels_size
        logits_batch = tf.reshape(logits, [-1, max_lengths, y_size])             # restoring to shape batch_size x max_length x labels_size

    with tf.variable_scope("loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_batch, labels=y)

        # exclude padding tokens from the gradient optimization
        mask = tf.sequence_mask(lengths)
        losses = tf.boolean_mask(losses, mask)

        loss = tf.reduce_mean(losses)

    with tf.variable_scope("softmax"):
        # return the label with the max score for each token
        preds = tf.cast(tf.argmax(logits_batch, axis=-1), tf.int32)

    with tf.variable_scope("optimizer"):
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return \
        x_1grams, x_2grams, x_3grams, y,\
        keep_pr, recurrent_keep_pr,\
        lengths, train,\
        loss, preds


def get_layered_model(
        pretrained_emb_1grams,
        pretrained_emb_2grams,
        hidden_size,
        layers,
        y_size,
        learning_rate
):
    """
    Multi-layer BiLSTM model based on 1-grams and 2-grams features.
    This model only differs from the baseline one from the number of Bi-LSTM layers.
    :param pretrained_emb_1grams: Embedding matrix as np.ndarray (for 1-grams)
    :param pretrained_emb_2grams: Embedding matrix as np.ndarray (for 2-grams)
    :param hidden_size: Output size of LSTM networks
    :param layers: Number of additional Bi-LSTM layers to be added (i.e. passing 1 implies having 2 layers in total)
    :param y_size: Size of the labels dictionary
    :param learning_rate: self-explanatory
    :return: Tensor operations for interaction with the built graph
    """
    x_1grams = tf.placeholder(tf.int32, shape=[None, None])         # shape: batch_size x max_length (for 1-grams)
    x_2grams = tf.placeholder(tf.int32, shape=[None, None])         # shape: batch_size x max_length (for 2-grams)
    y = tf.placeholder(tf.int32, shape=[None, None])                # shape: batch_size x max_length

    keep_pr = tf.placeholder(tf.float32, shape=[])
    recurrent_keep_pr = tf.placeholder(tf.float32, shape=[])

    # max_length for each sequence
    lengths = tf.placeholder(tf.int32, shape=[None])       # shape batch_size x 1

    # each sentence is wrapped by <S> and </S> tags
    lengths = lengths + 2

    with tf.variable_scope("embeddings"):
        # memory-efficient pre-trained embeddings loading
        embedding_matrix_1grams = tf.get_variable(                                  # shape: vocab_size x embedding_size (same size of the given embedding matrix)
            name="embeddings_1grams",
            shape=[pretrained_emb_1grams.shape[0], pretrained_emb_1grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_1grams),             # initializing to pre-trained_embeddings
            trainable=False                                                         # and making them a constant
        )

        embeddings_1grams = tf.nn.embedding_lookup(embedding_matrix_1grams, x_1grams)       # shape: batch_size x max_length x embedding_size

        # same treatment for 2-grams
        embedding_matrix_2grams = tf.get_variable(
            name="embeddings_2grams",
            shape=[pretrained_emb_2grams.shape[0], pretrained_emb_2grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_2grams),
            trainable=False
        )

        embeddings_2grams = tf.nn.embedding_lookup(embedding_matrix_2grams, x_2grams)

        # concatenate embeddings from 1-grams and 2-grams
        embeddings = tf.concat([embeddings_2grams, embeddings_1grams], axis=-1)             # shape: batch_size x max_length x embedding_size1 + embedding_size2

    with tf.variable_scope("bi-lstm"):
        # forward cell
        ltr_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)         # output shape: batch_size x max_length x hidden_size
        ltr_cell = tf.nn.rnn_cell.DropoutWrapper(
            ltr_cell,
            input_keep_prob=keep_pr,
            output_keep_prob=keep_pr,
            state_keep_prob=recurrent_keep_pr
        )

        # backward cell
        rtl_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)         # output shape: batch_size x max_length x hidden_size
        rtl_cell = tf.nn.rnn_cell.DropoutWrapper(
            rtl_cell,
            input_keep_prob=keep_pr,
            output_keep_prob=keep_pr,
            state_keep_prob=recurrent_keep_pr
        )

        # actual bi-lstm for layer 0 (baseline model)
        (ltr_outputs, rtl_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
            ltr_cell,
            rtl_cell,
            embeddings,
            sequence_length=lengths,
            dtype=tf.float32
        )

        # concat ltr_outputs and rtl_outputs to obtain the overall representation
        lstm_outputs = tf.concat([ltr_outputs, rtl_outputs], axis=-1)   # shape: batch_size x max_length x 2 * hidden_size

        if layers > 0:
            for layer in range(layers):

                # avoid parameter sharing by defining new variable scopes for each layer
                with tf.variable_scope("layer_%d" % (layer + 1), reuse=tf.AUTO_REUSE):

                    # forward cell
                    layer_ltr_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)           # output shape: batch_size x max_length x hidden_size
                    layer_ltr_cell = tf.nn.rnn_cell.DropoutWrapper(
                        layer_ltr_cell,
                        input_keep_prob=keep_pr,
                        output_keep_prob=keep_pr,
                        state_keep_prob=recurrent_keep_pr
                    )

                    # backward cell
                    layer_rtl_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)           # output shape: batch_size x max_length x hidden_size
                    layer_rtl_cell = tf.nn.rnn_cell.DropoutWrapper(
                        layer_rtl_cell,
                        input_keep_prob=keep_pr,
                        output_keep_prob=keep_pr,
                        state_keep_prob=recurrent_keep_pr
                    )

                    # actual bi-lstm for this layer takes the previous layer's outputs, concatenated
                    (ltr_outputs, rtl_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                        layer_ltr_cell,
                        layer_rtl_cell,
                        lstm_outputs,
                        sequence_length=lengths,
                        dtype=tf.float32
                    )

                    # concat ltr_outputs and rtl_outputs to obtain the overall representation for this layer
                    lstm_outputs = tf.concat([ltr_outputs, rtl_outputs], axis=-1)       # shape: batch_size x max_length x 2 * hidden_size

    with tf.variable_scope("dense"):
        W = tf.get_variable(                # shape: 2 * hidden_size x labels_size
            name="W",
            shape=[2 * hidden_size, y_size],
            dtype=tf.float32
        )

        b = tf.get_variable(                # shape: labels_size x 1
            name="b",
            shape=[y_size],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )

        max_lengths = tf.shape(lstm_outputs)[1]                                  # save the max_length of each sentence for later

        lstm_outputs_flat = tf.reshape(lstm_outputs, [-1, 2 * hidden_size])      # flattening to shape batch_size x 2 * hidden_size
        logits = lstm_outputs_flat @ W + b                                       # shape: batch_size x labels_size
        logits_batch = tf.reshape(logits, [-1, max_lengths, y_size])             # restoring to shape batch_size x max_length x labels_size

    with tf.variable_scope("loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_batch, labels=y)

        # exclude padding tokens from the gradient optimization
        mask = tf.sequence_mask(lengths)
        losses = tf.boolean_mask(losses, mask)

        loss = tf.reduce_mean(losses)

    with tf.variable_scope("softmax"):
        # return the label with the max score for each token
        preds = tf.cast(tf.argmax(logits_batch, axis=-1), tf.int32)

    with tf.variable_scope("optimizer"):
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return \
        x_1grams, x_2grams, y,\
        keep_pr, recurrent_keep_pr,\
        lengths, train,\
        loss, preds


def get_3grams_layered_model(
        pretrained_emb_1grams,
        pretrained_emb_2grams,
        pretrained_emb_3grams,
        hidden_size,
        layers,
        y_size,
        learning_rate
):
    """
    Multi-layer BiLSTM model based on 1-grams, 2-grams, and 3-grams features.
    This model combines the 3-grams features addition and the the multi-layer Bi-LSTM structure.
    :param pretrained_emb_1grams: Embedding matrix as np.ndarray (for 1-grams)
    :param pretrained_emb_2grams: Embedding matrix as np.ndarray (for 2-grams)
    :param pretrained_emb_3grams: Embedding matrix as np.ndarray (for 3-grams)
    :param hidden_size: Output size of LSTM networks
    :param layers: Number of additional Bi-LSTM layers to be added (i.e. passing 1 implies having 2 layers in total)
    :param y_size: Size of the labels dictionary
    :param learning_rate: self-explanatory
    :return: Tensor operations for interaction with the built graph
    """
    x_1grams = tf.placeholder(tf.int32, shape=[None, None])  # shape: batch_size x max_length (for 1-grams)
    x_2grams = tf.placeholder(tf.int32, shape=[None, None])  # shape: batch_size x max_length (for 2-grams)
    x_3grams = tf.placeholder(tf.int32, shape=[None, None])  # shape: batch_size x max_length (for 3-grams)
    y = tf.placeholder(tf.int32, shape=[None, None])  # shape: batch_size x max_length

    keep_pr = tf.placeholder(tf.float32, shape=[])
    recurrent_keep_pr = tf.placeholder(tf.float32, shape=[])

    # max_length for each sequence
    lengths = tf.placeholder(tf.int32, shape=[None])  # shape batch_size x 1

    # each sentence is wrapped by <S> and </S> tags
    lengths = lengths + 2

    with tf.variable_scope("embeddings"):
        # memory-efficient pre-trained embeddings loading
        embedding_matrix_1grams = tf.get_variable(
            # shape: vocab_size x embedding_size (same size of the given embedding matrix)
            name="embeddings_1grams",
            shape=[pretrained_emb_1grams.shape[0], pretrained_emb_1grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_1grams),  # initializing to pre-trained_embeddings
            trainable=False  # and making them a constant
        )

        embeddings_1grams = tf.nn.embedding_lookup(embedding_matrix_1grams, x_1grams)  # shape: batch_size x max_length x embedding_size

        # same treatment for 2-grams
        embedding_matrix_2grams = tf.get_variable(
            name="embeddings_2grams",
            shape=[pretrained_emb_2grams.shape[0], pretrained_emb_2grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_2grams),
            trainable=False
        )

        embeddings_2grams = tf.nn.embedding_lookup(embedding_matrix_2grams, x_2grams)

        # same treatment for 3-grams
        embedding_matrix_3grams = tf.get_variable(
            name="embeddings_3grams",
            shape=[pretrained_emb_3grams.shape[0], pretrained_emb_3grams.shape[1]],
            initializer=tf.constant_initializer(pretrained_emb_3grams),
            trainable=False
        )

        embeddings_3grams = tf.nn.embedding_lookup(embedding_matrix_3grams, x_3grams)

        # concatenate embeddings from 1-grams and 2-grams
        embeddings = tf.concat([embeddings_3grams, embeddings_2grams, embeddings_1grams], axis=-1)  # shape: batch_size x max_length x embedding_size1 + embedding_size2 + embedding_size3

    with tf.variable_scope("bi-lstm"):
        # forward cell
        ltr_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)         # output shape: batch_size x max_length x hidden_size
        ltr_cell = tf.nn.rnn_cell.DropoutWrapper(
            ltr_cell,
            input_keep_prob=keep_pr,
            output_keep_prob=keep_pr,
            state_keep_prob=recurrent_keep_pr
        )

        # backward cell
        rtl_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)         # output shape: batch_size x max_length x hidden_size
        rtl_cell = tf.nn.rnn_cell.DropoutWrapper(
            rtl_cell,
            input_keep_prob=keep_pr,
            output_keep_prob=keep_pr,
            state_keep_prob=recurrent_keep_pr
        )

        # actual bi-lstm for layer 0 (baseline model)
        (ltr_outputs, rtl_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
            ltr_cell,
            rtl_cell,
            embeddings,
            sequence_length=lengths,
            dtype=tf.float32
        )

        # concat ltr_outputs and rtl_outputs to obtain the overall representation
        lstm_outputs = tf.concat([ltr_outputs, rtl_outputs], axis=-1)   # shape: batch_size x max_length x 2 * hidden_size

        if layers > 0:
            for layer in range(layers):

                # avoid parameter sharing by defining new variable scopes for each layer
                with tf.variable_scope("layer_%d" % (layer + 1), reuse=tf.AUTO_REUSE):

                    # forward cell
                    layer_ltr_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)           # output shape: batch_size x max_length x hidden_size
                    layer_ltr_cell = tf.nn.rnn_cell.DropoutWrapper(
                        layer_ltr_cell,
                        input_keep_prob=keep_pr,
                        output_keep_prob=keep_pr,
                        state_keep_prob=recurrent_keep_pr
                    )

                    # backward cell
                    layer_rtl_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)           # output shape: batch_size x max_length x hidden_size
                    layer_rtl_cell = tf.nn.rnn_cell.DropoutWrapper(
                        layer_rtl_cell,
                        input_keep_prob=keep_pr,
                        output_keep_prob=keep_pr,
                        state_keep_prob=recurrent_keep_pr
                    )

                    # actual bi-lstm for this layer takes the previous layer's outputs, concatenated
                    (ltr_outputs, rtl_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                        layer_ltr_cell,
                        layer_rtl_cell,
                        lstm_outputs,
                        dtype=tf.float32
                    )

                    # concat ltr_outputs and rtl_outputs to obtain the overall representation for this layer
                    lstm_outputs = tf.concat([ltr_outputs, rtl_outputs], axis=-1)       # shape: batch_size x max_length x 2 * hidden_size

    with tf.variable_scope("dense"):
        W = tf.get_variable(                # shape: 2 * hidden_size x labels_size
            name="W",
            shape=[2 * hidden_size, y_size],
            dtype=tf.float32
        )

        b = tf.get_variable(                # shape: labels_size x 1
            name="b",
            shape=[y_size],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )

        max_lengths = tf.shape(lstm_outputs)[1]                                  # save the max_length of each sentence for later

        lstm_outputs_flat = tf.reshape(lstm_outputs, [-1, 2 * hidden_size])      # flattening to shape batch_size x 2 * hidden_size
        logits = lstm_outputs_flat @ W + b                                       # shape: batch_size x labels_size
        logits_batch = tf.reshape(logits, [-1, max_lengths, y_size])             # restoring to shape batch_size x max_length x labels_size

    with tf.variable_scope("loss"):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_batch, labels=y)

        # exclude padding tokens from the gradient optimization
        mask = tf.sequence_mask(lengths)
        losses = tf.boolean_mask(losses, mask)

        loss = tf.reduce_mean(losses)

    with tf.variable_scope("softmax"):
        # return the label with the max score for each token
        preds = tf.cast(tf.argmax(logits_batch, axis=-1), tf.int32)

    with tf.variable_scope("optimizer"):
        train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return \
        x_1grams, x_2grams, x_3grams, y,\
        keep_pr, recurrent_keep_pr,\
        lengths, train,\
        loss, preds
