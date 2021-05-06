import spacy
import spacy.cli
import getting_tweets_v2
import re

""" Spacy loading """
# spacy.cli.download('de_core_news_md')
#
# nlp_ge = spacy.load('de_core_news_md')

""" Make files -> preparation for further processing """

getting_tweets_v2.collect_tweets_in_files()
# tweet_file = open('tweets_2.txt', 'r').read()


""" Remove emoticons, usernames, retweets-definition -> plain text """

""" Make spacy doc for text processing """
