import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')


def read_csv_files():
    """This method reads training data for three personalities from files
    Return: three DataFrame objects, each contains data for one personality
    """
    df_1 = pd.read_csv('../data/ms_chatbot_dialogs/qna_chitchat_the_comic.tsv', sep="\t")
    df_2 = pd.read_csv('../data/ms_chatbot_dialogs/qna_chitchat_the_friend.tsv', sep="\t")
    df_3 = pd.read_csv('../data/ms_chatbot_dialogs/qna_chitchat_the_professional.tsv', sep="\t")
    return df_1, df_2, df_3


def prepare_data_by_dataframe(df):
    """Preprocessing method
    includes word tokenizating, lowercasing, punctuation removal, lemmatization, stop-word removal
    Param df: DataFrame objects contains data for one personality
    Return: dict object contains (question, answer) pair, {answer: index} dict and all other information for training
    """
    max_input_words_amount = 0
    lemmatizer = WordNetLemmatizer()
    training_data = []
    whole_answers = set()

    for index, row in df.iterrows():
        question = row['Question'].strip()
        answer = row['Answer'].strip()
        whole_answers.add(answer)

        token_question = set()
        # tokenize
        for word in word_tokenize(question):
            # convert to lower case
            word = word.lower()
            # remove punctuations
            word = re.sub(r'[^\w\s]', '', word)
            # remove stopwords
            if word not in stopwords.words('english') and len(word) > 0:
                # lemmatize
                word = lemmatizer.lemmatize(word)
                token_question.add(word)

        max_input_words_amount = max(len(token_question), max_input_words_amount)
        training_data.append((list(token_question), answer))

    whole_answers = list(whole_answers)
    whole_answers.append('_U_')  # unknown
    answer_dict = {a: i for i, a in enumerate(whole_answers)}

    return {
        'training_data': training_data,  # a list stores (tokenized_question, answer) pair
        'whole_answers': whole_answers,  # a list stores all answers
        'answer_dict': answer_dict,  # a dict stores all answers and their related index
        'max_input_words_amount': max_input_words_amount
    }


def build_dataset():
    """Return a dict contains all data to be used for training three chatbots
    """
    all_df = read_csv_files()
    dataset = {
        'comic': prepare_data_by_dataframe(all_df[0]),
        'friend': prepare_data_by_dataframe(all_df[1]),
        'professional': prepare_data_by_dataframe(all_df[2])
    }
    return dataset
