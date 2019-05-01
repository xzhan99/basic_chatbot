import logging

import tensorflow as tf

from configuration import WORD_DiMENSION


def build_seq2seq_model(n_class):
    # Setting Hyperparameters
    learning_rate = 0.002
    n_hidden = 128
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
        enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
        outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)
    # Decoder Cell
    with tf.variable_scope('decode'):
        dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
        dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
        outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                                initial_state=enc_states,
                                                dtype=tf.float32)

    model = tf.layers.dense(outputs, n_class, activation=None)
    # compute loss by MSE
    cost = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(  # the final result is sparse
            logits=model, labels=targets))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return enc_input, dec_input, targets, model, cost, optimizer


def train_seq2seq_model(enc_input, dec_input, targets, cost, optimizer, input_batch, output_batch, target_batch):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    total_epoch = 200

    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,  # shape(5, 100)
                                      dec_input: output_batch,  # shape(1, 98)
                                      targets: target_batch})  # shape(1, 1)
        logging.debug('Epoch: {:3d}\tcost = {:.6f}'.format(epoch + 1, loss))

    logging.debug('Epoch: {:3d}\tcost = {:.6f}'.format(epoch + 1, loss))
    return sess


def save_seq2seq_model(sess, path):
    # save seq model to google drive
    saver = tf.train.Saver()
    saver.save(sess, path)


def load_seq2seq_model(path):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # restore (Load) seq model from google drive
    saver = tf.train.Saver()
    saver.restore(sess, path)
    return sess
