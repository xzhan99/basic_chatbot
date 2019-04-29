from gensim.models import FastText
import re

from nltk import WordNetLemmatizer

WORD_DiMENSION = 100  # the amount of float number in every vector


def read_sentences(path):
    """
    This method extracts and preprocesses sentences that read from file
    Param path: the path of file
    Return: a list that includes well preprocessed sentences
    """
    lemmatizer = WordNetLemmatizer()
    lines = ''
    with open(path, encoding='iso-8859-1') as file:
        lines = file.readlines()

    token_sentences = []
    for line in lines:
        line = line.split('+++$+++')[-1].split()  # the file uses '+++$+++' to separate every column
        token_words = []
        for word in line:
            # convert to lowercase
            word = word.lower()
            # lemmatize
            word = lemmatizer.lemmatize(word)
            # remove punctuations
            word = re.sub(r'[^\w\s]', '', word)
            # remove empty sentence
            if len(word) > 0:
                token_words.append(word)
        token_sentences.append(token_words)
    return token_sentences


def load_word_embeddings_model(path):
    # load fasttext model
    return FastText.load(path)
