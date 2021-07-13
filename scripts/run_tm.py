import pickle
import gensim
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from preprocessing_tweets import ngrams, make_bow_corpus
from pprint import pprint

"""Load our tweet dictionary"""

tweet_dictionary = Dictionary.load('./../data/tweet_dictionary')
with open('../data/docs', 'rb') as f:
    docs = pickle.load(f)


""" Ngrams """
ngram_docs = ngrams(input_docs=docs)


"""Make BOW representation of our corpus """

corpus = make_bow_corpus(tweet_dictionary, ngram_docs)

""" Save BOW corpus """

with open('../data/bow_corpus', 'wb') as f:
    pickle.dump(corpus, f)

print('Number of unique tokens: %d' % len(tweet_dictionary))
print('Number of documents: %d' % len(corpus))

"""Set training parameters."""
num_topics = 5  # Number of topics, here relatively low so we can interpret them more easily -> can be set higher
chunk_size = 7  # Numbers of documents fed into the training algorithm (we have 7)
passes = 25  # Number of times trained on the entire corpus
iterations = 60  # Number of loops over each document
eval_every = None  # Don't evaluate model perplexity, takes too much time.

""" Make a index to word dictionary."""
temp = tweet_dictionary[0]  # This is only to "load" the dictionary.
id2word = tweet_dictionary.id2token

"""Create model
We set alpha = 'auto' and eta = 'auto'. Again this is somewhat technical, but essentially we are automatically learning
two parameters in the model that we usually would have to specify explicitly."""
model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunk_size,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

""" Save model so we can load it later - only needed if you need to train the model from anew """
model_file = '.././data/model/LDA_model_v1'
model.save(model_file)

""" Tests """
# Top topics
top_topics = model.top_topics(corpus)  # , num_words=20) Default value = 20, input is our corpus in BOW format

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
"""Topic Coherence measures score a single topic by measuring the degree of semantic similarity between high scoring 
words in the topic. These measurements help distinguish between topics that are semantically interpretable topics and 
topics that are artifacts of statistical inference """
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

pprint(top_topics)

""" Things to experiment with: 
1. no_above and no_below parameters in filter_extremes method.
2. Adding bi-, trigrams or even higher order n-grams.
3. Consider whether using a hold-out set or cross-validation is the way to go for you."""
