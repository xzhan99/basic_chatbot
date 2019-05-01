from gensim.models import FastText

from configuration import ROOT_PATH, WORD_DiMENSION
from word_embeddings.skipgram_model import read_sentences

if __name__ == '__main__':
    sentences = read_sentences(ROOT_PATH + '/data/cornell_movie_dialogs_corpus/movie_lines.txt')

    sg_model = FastText(size=WORD_DiMENSION, window=5, min_count=10, workers=4, sg=1)
    sg_model.build_vocab(sentences)
    sg_model.train(sentences, total_examples=sg_model.corpus_count, epochs=1)
    sg_model.save(ROOT_PATH + '/models/embeddings/gensim_fasttext.models')
