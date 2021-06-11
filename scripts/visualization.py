import pickle
import gensim
import pyLDAvis.gensim_models
import pyLDAvis
from gensim.models import LdaModel

""" Visualization method, call with input parameters """


def visualize_data(bow_corpus, tweet_dictionary, lda_model):
    lda_visualization = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, tweet_dictionary)
    pyLDAvis.save_html(lda_visualization, 'vis.html')


""" Load corpus, dictionary  """
with open('../data/bow_corpus', 'rb') as input_file:
    bow_corpus_input = pickle.load(input_file)

tweet_dictionary_input = gensim.corpora.Dictionary.load('../data/tweet_dictionary')

""" Load model """
model = LdaModel.load('../data/model/LDA_model_v1')

""" TEST """
visualize_data(bow_corpus_input, tweet_dictionary_input, model)
