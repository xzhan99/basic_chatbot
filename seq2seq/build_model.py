import logging

import tensorflow as tf

from configuration import WORD_DiMENSION, LEARNING_RATE, N_HIDDEN, OUTPUT_KEEP_PROB, TOTAL_EPOCHS


def build_seq2seq_model(n_class):
    # Setting Hyperparameters
    n_input = WORD_DiMENSION  # n_input equals to the amout of float numbers in a word embedding vector

    # Neural Network Model
    tf.reset_default_graph()

    # encoder/decoder shape = [batch size, time steps, input size]
    enc_input = tf.placeholder(tf.float32, [None, None, n_input])
    dec_input = tf.placeholder(tf.float32, [None, None, n_class])
    # target shape = [batch size, time steps]
    targets = tf.placeholder(tf.int64, [None, None])

    # Encoder Cell
    with tf.variable_scope('encode'):
        enc_cell = tf.nn.rnn_cell.BasicRNNCell(N_HIDDEN)
        enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=OUTPUT_KEEP_PROB)
        outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)
    # Decoder Cell
    with tf.variable_scope('decode'):
        dec_cell = tf.nn.rnn_cell.BasicRNNCell(N_HIDDEN)
        dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=OUTPUT_KEEP_PROB)
        outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                                initial_state=enc_states,
                                                dtype=tf.float32)
    model = tf.keras.layers.Dense(n_class, activation=None)(outputs)
    # compute loss by MSE
    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(  # the final result is sparse
            logits=model, labels=targets))

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    return enc_input, dec_input, targets, model, cost, optimizer


def train_seq2seq_model(enc_input, dec_input, targets, cost, optimizer, input_batch, output_batch, target_batch):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(TOTAL_EPOCHS):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,  # shape(5, 100)
                                      dec_input: output_batch,  # shape(1, 98)
                                      targets: target_batch})  # shape(1, 1)
        logging.debug('Epoch: {:3d}\tcost = {:.6f}'.format(epoch + 1, loss))

    logging.debug('Epoch: {:3d}\tcost = {:.6f}'.format(epoch + 1, loss))
    return sess


def save_seq2seq_model(sess, path):
    # save seq models to google drive
    saver = tf.train.Saver()
    saver.save(sess, path)


def load_seq2seq_model(path):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # restore (Load) seq models from google drive
    saver = tf.train.Saver()
    saver.restore(sess, path)
    return sess
