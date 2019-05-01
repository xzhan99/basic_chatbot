import logging

from chatbot import ROOT_PATH
from configuration import ALL_PERSONALITY, LOGGING_FORMAT
from seq2seq.apply_embedding_model import make_batch
from seq2seq.build_model import train_seq2seq_model, save_seq2seq_model, build_seq2seq_model
from seq2seq.preprocessing import build_dataset, save_preprocessed_data
from word_embeddings.skipgram_model import load_word_embeddings_model

logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT)


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
    # data preprocessing
    logging.info('preforming data preprocessing')
    dataset = build_dataset()
    save_preprocessed_data(dataset, ROOT_PATH + '/data/preprocessed_data.json')

    sg_model = load_word_embeddings_model(ROOT_PATH + '/models/embeddings/gensim_fasttext.models')
    for p in ALL_PERSONALITY:
        logging.info('training models for %s' % p)
        train_and_save_session(p, sg_model, ROOT_PATH + '/models/seq2seq/%s_model.ckpt' % p)
