import logging
import re
import warnings

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

from configuration import ROOT_PATH, ALL_PERSONALITY
from seq2seq.apply_embedding_model import make_batch
from seq2seq.build_model import load_seq2seq_model, build_seq2seq_model
from seq2seq.preprocessing import load_preprocessed_data
from word_embeddings.skipgram_model import load_word_embeddings_model
import tensorflow as tf

warnings.simplefilter(action='ignore', category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_models():
    """
    Load word embedding and seq2seq model
    return: sg_model, sessions
    """
    # use above function to load embedding model
    sg_model = load_word_embeddings_model(ROOT_PATH + '/model/embeddings/gensim_fasttext.model')
    # use above function to load three sequence models
    sessions = {}
    for personality in ALL_PERSONALITY:
        sessions[personality] = load_seq2seq_model(ROOT_PATH + '/model/seq2seq/%s_model.ckpt' % personality)
    return sg_model, sessions


def preprocess(sentence):
    """
    This method preforms preprocessing on question
    Param sentence: question string
    Return: tokenized question
    """
    lemmatizer = WordNetLemmatizer()
    token_question = set()
    # tokenize
    for word in word_tokenize(sentence):
        # convert to lower case
        word = word.lower()
        # remove punctuations
        word = re.sub(r'[^\w\s]', '', word)
        # remove stopwords
        if word not in stopwords.words() and len(word) > 0:
            # stem
            word = lemmatizer.lemmatize(word)
            token_question.add(word)
    return token_question


def generate_pair_data(tokens, personality):
    """
    This method aims to generate the same form of data with training method for prediction
    Param tokens: tokenized question
    Param personality: personality
    """
    seq_data = [[list(tokens), '_U_']]  # one more pair of brackets since there is only one senctence during prediction
    answer_dict = dataset[personality]['answer_dict']
    max_input_words_amount = dataset[personality]['max_input_words_amount']
    return {
        'training_data': seq_data,  # (question, answer) pair
        'answer_dict': answer_dict,  # {answer string: index}
        'max_input_words_amount': max_input_words_amount
    }


def decode_answer(result, personality):
    # decode, convert index number to actual word
    decoded = [dataset[personality]['whole_answers'][i] for i in result[0]]
    return ' '.join(decoded)


def change_personality(command, current_personality):
    # regex pattern designed to match personality changing command
    matched = re.findall(r'^set_personality (.+?)\.?$', command)
    if len(matched) < 1:
        return 0, None  # return error code 0 when command pattern doesn't match
    # for the given command, it may contains several parts to be matched, here only keeps the first one
    personality = matched[0].strip()
    if personality in ALL_PERSONALITY:
        if personality == current_personality:
            return 1, personality  # return error code 1 when new personality is same to the current one
        return 2, personality  # return error code 2 when personality can be successfully changed
    else:
        return 3, personality  # return error code 3 when personality is not pretrained


def end_chatting(command):
    if len(command) >= 3 and 'bye' == command[:3]:
        return True
    return False


def answer(sess, embedding_model, personality, sentence):
    if personality not in ALL_PERSONALITY:
        return 'Personality \'%s\' dose not exist.' % personality

    token_question = preprocess(sentence)
    # prepare data for prediction
    personality_data = generate_pair_data(token_question, personality)
    input_batch, output_batch, target_batch = make_batch(personality_data,
                                                         embedding_model)
    prediction = tf.argmax(model, 2)
    # predict result, it contains 2 int digits
    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
    translated = decode_answer(result, personality)
    return translated


if __name__ == '__main__':
    # load preprocessed data
    dataset = load_preprocessed_data(ROOT_PATH + '/data/preprocessed_data.json')

    # initial seq2seq parameters
    enc_input, dec_input, targets, model, cost, optimizer = build_seq2seq_model(
        len(dataset[ALL_PERSONALITY[0]]['answer_dict']))
    sg_model, sessions = load_models()  # read embeddings and seq model from google drive

    current_personality = 'professional'  # default personality
    question = ''

    while True:
        question = input('User: ').strip().lower()  # convert question to lowercase
        if len(question) == 0:
            print('Chatbot: ', 'Invalid input!')
            continue
        # change personality if command is matched
        code, new_personality = change_personality(question, current_personality)
        # if personality dose not exist, it would not be changed
        if code:
            if code == 1:
                response = 'The given personality is same to the current.'
            elif code == 3:
                response = 'Personality \'%s\' is not available.' % new_personality
            elif code == 2:
                response = 'Personality is changed to %s.' % new_personality
                current_personality = new_personality
            else:
                response = 'Error detected during changing personality.'
            print('Chatbot: ', response)
            continue
        # predict answer
        response = answer(sessions[current_personality], sg_model, current_personality, question)
        print('Chatbot: ', response)

        # finish chatting if command is matched
        if end_chatting(question):
            break
