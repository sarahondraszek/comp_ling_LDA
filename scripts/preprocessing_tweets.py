import glob
import pickle

from gensim.models import Phrases
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

spacy.cli.download('de_core_news_md')
nlp = spacy.load('de_core_news_md')
nlp.max_length = 2000000


def preprocess_data(input_file_list):
    """
    Preprocessing the raw data. Removing 'unnecessary' symbols, words etc. from the tweet corpus,
    setting it to lower case, removing stopwords and words with a length < 2 and tokenizing it.

    :param input_file_list: List of files (tweet corpus)
    :return: List of documents that are each represented as lists of tokens
    """
    stopwords_ger = nltk.corpus.stopwords.words("german")  # Liste der Stoppwörter
    temp_docs = []
    for file in input_file_list:
        print('Working on:' + file)
        dataframe = pd.read_csv(file, usecols=['text'], lineterminator='\n')  # Datei öffnen
        datastring = dataframe.to_string(index=False)  # String erstellen aus Dataframe
        datastring = re.sub(r"@[\S]*|#", "", datastring)  # Nutzernamen und Hashtag-Zeichen entfernen
        datastring = re.sub(r"https:[\S]*|www[\S]*", "", datastring)  # Links entfernen
        datastring = re.sub(r"[0-9]*", "", datastring)  # Zahlen entfernen
        datastring = re.sub(r"\\n", " ", datastring)  # Zeilenumbruchmarker entfernen
        datastring = re.sub(r"\W", " ", datastring)  # Satzzeichen entfernen
        datastring = re.sub(r"RT|NaN|via", "", datastring)  # Entfernen von twitterspezifiscehen Markern

        token = nltk.wordpunct_tokenize(datastring.lower())  # Tokeniserung + nur Kleinschreibung

        removed_stopwords = [w for w in token if not w in stopwords_ger]  # Stoppwörter entfernen
        text = [w for w in removed_stopwords if len(w) > 2]  # Wörter < 2 Buchstaben entfernen

        list = []  # Liste der Token wird erzeugt und der Liste docs hinzugefügt
        for w in text:
            list.append(w)
        temp_docs.append(list)

    return temp_docs


def lemmatizer(input_docs):
    """
    Lemmatizing words

    :param input_docs: List of documents (represented as list of tokens)
    :return: List of documents (represented as list of lemmata)
    """
    lemmatized_words = []
    for word_list in input_docs:
        temp_list = []
        temp_doc = nlp(' '.join(word_list))
        for word in temp_doc:
            temp_list.append(word.lemma_)
        lemmatized_words.append(temp_list)
    return lemmatized_words


def ngrams(input_docs):
    """
    Add bigrams (and possibly trigrams) to docs (only ones that appear 20 times or more).
    Uncomment trigram lines for trigram addition.

    :param input_docs: input docs file (gensim format)
    :return: docs file (list of lists) with appended ngrams
    """
    output_docs = input_docs
    bigram = Phrases(output_docs, min_count=20)
    # trigram = Phrases(bigram[output_docs], min_count=20)

    for idx in range(len(output_docs)):
        for bigram_ in bigram[output_docs[idx]]:
            if '' in bigram_:
                # Token is a bigram, add to document.
                output_docs[idx].append(bigram_)
                # for token in trigram[bigram[bigram_]]:
                #     if '' in token:
                #         output_docs[idx].append(token)
    return output_docs


def make_bow_corpus(input_dictionary, input_docs):
    """
    BOW representation

    :param input_dictionary: Loaded dictionary from storage as input
    :param input_docs: Loaded docs (list of documents as list of lemmata) from storage
    :return: Bag-Of-Words representation of documents
    """
    return [input_dictionary.doc2bow(doc) for doc in input_docs]


print(time.process_time() - start)
