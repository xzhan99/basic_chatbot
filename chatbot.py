import re

from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

from seq2seq.apply_embedding_model import make_batch
from seq2seq.build_model import load_seq2seq_model, enc_input, dec_input, targets
from word_embeddings.skipgram_model import load_word_embeddings_model
import tensorflow as tf

# all personality opinions
ALL_PERSONALITY = ('professional', 'friend', 'comic')


def load_models():
    """
    Load word embedding and seq2seq model
    return: sg_model, sessions
    """
    # use above function to load embedding model
    sg_model = load_word_embeddings_model()
    # use above function to load three sequence models
    sessions = {}
    for personality in ALL_PERSONALITY:
        sessions[personality] = load_seq2seq_model('model_%s.ckpt' % personality)
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


# Answer the question using the trained model
def answer(sess, sentence, personality, skipgram):
    if personality not in ALL_PERSONALITY:
        return 'Personality \'%s\' dose not exist.' % personality

    token_question = preprocess(sentence)
    # prepare data for prediction
    personality_data = generate_pair_data(token_question, personality)
    input_batch, output_batch, target_batch = make_batch(personality_data,
                                                         skipgram)  # make_batch() is defined in section2
    prediction = tf.argmax(model, 2)
    # predict result, it contains 2 int digits
    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
    translated = decode_answer(result, personality)
    return translated
