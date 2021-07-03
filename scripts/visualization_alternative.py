import pickle
import gensim
import pyLDAvis.gensim
import pyLDAvis
from gensim.models import LdaModel
""" Load corpus, dictionary  """

with open('../data/bow_corpus', 'rb') as input_file:
    corpus = pickle.load(input_file)

tweet_dictionary = gensim.corpora.Dictionary.load('../data/tweet_dictionary')

""" Load model """
model = LdaModel.load('../data/model/LDA_model_v1')

""" Visualization """

lda_visualization = pyLDAvis.gensim.prepare(model, corpus, tweet_dictionary)
pyLDAvis.show(lda_visualization)
