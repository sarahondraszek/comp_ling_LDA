import searchtweets
from searchtweets import load_credentials
from searchtweets import collect_results
from searchtweets import ResultStream
from searchtweets import gen_request_parameters
import csv
import pandas
from pathlib import Path

""" This script is for scraping recent tweets (past 7 days) from twitter directly. 
Based on: https://pypi.org/project/searchtweets-v2/#description """

# Where to save our results should be defined here
saving_path = r'/Volumes/My Passport for Mac/tweets/tweets_metadata_08052021.csv'

""" Credentials file for developer accounts, mandatory for the access to API!!! """

credentials = load_credentials(filename="credentials.yaml",
                               yaml_key="credentials",
                               env_overwrite=False)  # change if needed

""" Query can be defined here, 
there always has to be a certain search keyword, I put 'a' here because of the wider 
reach, might be possible to exclude (have to do further investigations in this case.
 results_per_call can be redefined via a .yaml file """

# request params for this query
query = searchtweets.gen_request_parameters("a lang:de", start_time="2021-05-08T00:00",
                                            end_time="2021-05-08T23:59",
                                            results_per_call=100)

""" List of tweet dicts, including the ids and the tweet text. Can be directly printed or stored in a file """

# tweets = collect_results(query,
#                          max_tweets=100,
#                          result_stream_args=credentials)
#
# with open('./tweets.txt', 'w') as tweet_file:
#     for x in tweets:
#         for y in x:
#             if y == 'text':
#                 tweet_file.write(x[y] + '\n')



def check_files():

    try:
        my_file = Path(saving_path)
        my_path = my_file.resolve(strict='True')
    except FileNotFoundError:
        return False
    else:
        return True



def collect_tweets_in_files():

    """ Using a ResultStream for getting tweets
      We can configure the amount of pages/tweets we want to obtain """

    if not check_files():  # file should not be already existing

        max_results = 10000
        max_pages = 300
        max_tweets = 15000

        rs = ResultStream(request_parameters=query,
                          max_results=max_results,
                          max_pages=max_pages,
                          **credentials)

        # Set how many tweets we want to catch
        rs.max_tweets = max_tweets

        tweets_2 = list(rs.stream())
        dataframe = pandas.DataFrame(tweets_2)

        csv_file = dataframe.to_csv(saving_path)
    else:
        print(FileExistsError, 'File already exists! Please check if you really want to overwrite the file.')

# with open('./tweets_2.txt', 'w') as tweet_file_2:
#     for x in tweets_2:
#         for y in x:
#             if y == 'text':
#                 tweet_file_2.write(x[y] + '\n')
