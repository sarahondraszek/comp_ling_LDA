from preprocessing_tweets import preprocess_data, lemmatizer
import glob
import pickle
from gensim.corpora import Dictionary

file_list = glob.glob('./corpus/*.csv')
docs = preprocess_data(file_list)

docs = lemmatizer(input_docs=docs)

""" Save docs list """

with open('docs', 'wb') as f:
    pickle.dump(docs, f)

""" Make gensim.Dictionary and filter extremes """

dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=3, no_above=0.9)

""" Save our dictionary to a file so we can load it in the LDA script """

dictionary.save('./tweet_dictionary')
