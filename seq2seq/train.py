import logging

from seq2seq.apply_embedding_model import make_batch
from seq2seq.build_model import train_seq2seq_model, save_seq2seq_model, build_seq2seq_model
from seq2seq.preprocessing import dataset
from word_embeddings.skipgram_model import load_word_embeddings_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALL_PERSONALITY = ('professional', 'friend', 'comic')


def generate_batch_data(personality, embedding_model):
    input_batch, output_batch, target_batch = make_batch(dataset[personality], embedding_model)
    return input_batch, output_batch, target_batch


def train_session(personality, embedding_model):
    input_batch, output_batch, target_batch = generate_batch_data(personality, embedding_model)
    enc_input, dec_input, targets, model, cost, optimizer = build_seq2seq_model(
        len(dataset[personality]['answer_dict']))
    session = train_seq2seq_model(enc_input, dec_input, targets, cost, optimizer, input_batch, output_batch,
                                  target_batch)
    return session


def train_and_save_session(personality, embedding_model, path):
    sess = train_session(personality, embedding_model)
    # save session
    save_seq2seq_model(sess, path)


if __name__ == '__main__':
    sg_model = load_word_embeddings_model('../model/embeddings/gensim_fasttext.model')
    for p in ALL_PERSONALITY:
        logging.info('start training model for %s' % p)
        train_and_save_session(p, sg_model, '../model/seq2seq/%s_model.ckpt' % p)
