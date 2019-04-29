import numpy as np
import tensorflow as tf

from word_embeddings.skipgram_model import WORD_DiMENSION


def get_vectors_q(sentence, max_amount, skipgram):
    """
    This method gets token index vector of questions and add paddings if the word is shorter than the maximum number of words
    Param sentence: tokenized question list
    Param max_amount: maximum number of words in a question
    Param skipgram: word embedding model
    Return: vectors of question
    """
    diff = max_amount - len(sentence)

    # add paddings if the word is shorter than the maximum number of words
    for x in range(diff):
        sentence.append('_P_')

    # convert tokens to index
    ids = np.ndarray(shape=(max_amount, WORD_DiMENSION))
    for i, word in enumerate(sentence):
        ids[i] = skipgram[word]
    return ids


def get_vector_a(index_dict, answer, one_hot=False):
    """
    This method converts answer string to index or one-hot encode data
    Param index_dict: question to index dictionary
    Param answer: answer string
    Param one_hot: return one-hot encoded value or not
    Return: vectors of answer
    """
    answer_dimension = len(index_dict)
    index = index_dict[answer]
    if one_hot:
        vector = [1 if i == index else 0 for i in range(answer_dimension)]  # one-hot encoding
        return np.array(vector).reshape(1, answer_dimension)  # convert to ndarray object
    else:
        return np.array(index).reshape(1)  # convert to ndarray object


def make_batch(personality_data, skipgram):
    """
    This method generates a batch data for training/testing
    Param personality_data: all required data for a personality, which is generated in section 1
    Param skipgram: word embedding model
    Return: input_batch, output_batch, target_batch
    """
    training_data = personality_data['training_data']
    answer_dict = personality_data['answer_dict']
    max_input_words_amount = personality_data['max_input_words_amount']

    input_batch = []
    output_batch = []
    target_batch = []
    for question, answer in training_data:
        # Input for encoder cell, convert question to vector
        input_data = get_vectors_q(question, max_input_words_amount, skipgram)
        # Input for decoder cell
        output_data = get_vector_a(answer_dict, answer, one_hot=True)  # one-hot encoding of answer
        # Output of decoder cell (Actual result)
        target = get_vector_a(answer_dict, answer)  # index of answer
        input_batch.append(input_data)
        output_batch.append(output_data)
        target_batch.append(target)

    return input_batch, output_batch, target_batch


# testing: Generate a batch data
input_batch, output_batch, target_batch = make_batch(dataset['professional'], sg_model)
print(input_batch[0].shape)
print(output_batch[0].shape)
print(target_batch[0].shape)
