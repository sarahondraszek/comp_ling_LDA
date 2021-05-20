import glob
import pickle

from nltk.util import pr
import pandas as pd
import re
import nltk
import spacy
import spacy.cli
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
import time

""" Time measurements """

start = time.process_time()

""" Spacy installments """

# spacy.cli.download('de_core_news_md')
nlp = spacy.load('de_core_news_md')
nlp.max_length = 2000000


def preprocess_data(file_list):
    docs = []
    for file in file_list:
        print('Working on:' + file)
        dataframe = pd.read_csv(file, usecols=['text'], lineterminator='\n')  # Datei öffnen
        datastring = dataframe.to_string(index=False)  # String erstellen aus Dataframe
        datastring = re.sub(r"@[\S]*|#", "", datastring)  # Nutzernamen und Hashtag-Zeichen entfernen
        datastring = re.sub("https:[\S]*|www[\S]*", "", datastring)  # Links entfernen
        datastring = re.sub(r"[0-9]*", "", datastring)  # Zahlen entfernen
        datastring = re.sub(r"\\n", " ", datastring)  # Zeilenumbruchmarker entfernen
        datastring = re.sub("\W", " ", datastring)  # Satzzeichen entfernen
        datastring = re.sub("RT|NaN|via", "", datastring)  # Entfernen von twitterspezifiscehen Markern

        token = nltk.wordpunct_tokenize(datastring.lower())  # Tokeniserung + nur Kleinschreibung

        stopwords_ger = nltk.corpus.stopwords.words("german")  # Liste der Stoppwörter
        removed_stopwords = [w for w in token if not w in stopwords_ger]  # Stoppwörter entfernen
        text = [w for w in removed_stopwords if len(w) > 2]  # Wörter < 2 Buchstaben entfernen

        list = []  # Liste der Token wird erzeugt und der Liste docs hinzugefügt
        for w in text:
            list.append(w)
        docs.append(list)

    return docs


""" Create docs from files (please remember to set file path correctly """

file_list = glob.glob('../data/corpus/*.csv')
docs = preprocess_data(file_list)

""" Lemmatizing words """


def lemmatizer(docs):
    lemmatized_words = []
    for word_list in docs:
        temp_list = []
        temp_doc = nlp(' '.join(word_list))
        for word in temp_doc:
            temp_list.append(word.lemma_)
        lemmatized_words.append(temp_list)
    return lemmatized_words


docs = lemmatizer(docs=docs)

""" Save docs list """
with open('docs', 'wb') as f:
    pickle.dump(docs, f)

""" Make gensim.Dictionary and filter extremes """

dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=3, no_above=0.5)

""" Save our dictionary to a file so we can load it in the LDA script """

dictionary.save('../data/tweet_dictionary')


""" BOW representation """


def make_bow_corpus(input_dictionary, docs):
    return [input_dictionary.doc2bow(doc) for doc in docs]


print(time.process_time() - start)
