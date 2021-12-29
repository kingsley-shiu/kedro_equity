#%%

# Import the necessary libraries ------> requests
import requests
from urllib.request import urlopen, Request

# Import the necessary libraries ------> BeautifulSoup
from bs4 import BeautifulSoup

# Import the necessary libraries ------> Pandas,numpy,matplotlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# Import the necessary libraries ------> nltk Module, sentiment analyser nltk vader
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Import the necessary libraries ------> sentiment analyser textblob
# from textblob import TextBlob
# from textblob import Word

# Import the necessary libraries ------> basic aricle summarization
from newspaper import Article
from yahoo_fin import news

# Import the necessary libraries ------> Deep Learning Summarization
# from transformers import PegasusTokenizer, PegasusForConditionalGeneration
# Import the necessary libraries ------> sentiment analyser Pegasus
# from transformers import pipeline

from tqdm.notebook import tqdm
tqdm.pandas()

#%%


def finviz_parser_data(ticker):

    url = 'https://finviz.com/quote.ashx?t={}'.format(ticker)
    # sending request for getting the html code of the Url
    try:
        request = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(request)

        #parsing the HTML with BeautifulSoup
        soup = BeautifulSoup(response, 'html')
        return soup
    except Exception as e:
        print(e)


def correct_time_formatting(time_data):
    date = []
    time = []
    for z in time_data:
        a = z.split(" ")
        if len(a) == 2:
            date.append(a[0])
            time.append(a[1])
        else:
            date.append("r")
            time.append(a[0])
    l = 0
    r = 1
    lister = []
    #print(l,r)
    while r < len(date):
        if len(date[r]) == 9:
            lister.append(date[l:r])
            #print(l,r)
            l = r
            #print(l,r)
        elif r == len(date)-1:
            r = len(date)
            #print(l,r)
            lister.append(date[l:r])
        r += 1
    n = 0
    while n < len(lister):

        lister[n] = [lister[n][0] for x in lister[n] if x == 'r' or x == lister[n][0]]
        n += 1
    final_time = []
    y = 0
    while y < len(lister):
        final_time += lister[y]
        y += 1
    count = 0
    time_correct = []
    while count < len(final_time):
        time_correct.append((final_time[count]+" "+time[count]))
        count += 1
    return time_correct


def finviz_create_write_data(soup, file_name=None):
    try:
        news_reporter_title = [row.text for row in soup.find_all(class_='news-link-right') if row is not None]
        #news_reporter_title
        news_reported = [row.text for row in soup.find_all(class_='news-link-left') if row is not None]
        #news_reported
        news_url = [row.find('a', href=True)["href"] for row in soup.find_all(class_='news-link-left') if row is not None]
        '''
        solution 2:
        atags = [row.find('a') for row in soup.find_all(class_ ='news-link-left') if row is not None]
        news_url = [link['href'] for link in atags]
        '''
        date_data = [row.text for row in soup.find_all('td', attrs={"width": "130", 'align': 'right'}) if row is not None]
        time = correct_time_formatting(date_data)
    except Exception as e:
        print(e)
    data = {"Time": time, 'News Reporter': news_reporter_title, "News Headline": news_reported, "URL": news_url}
    finviz_news_df = pd.DataFrame.from_dict(data)
    if file_name:
        finviz_news_df.to_csv(file_name + '_finviz_stock.csv')
        print(file_name + "_finviz_stock.csv is created")
    return finviz_news_df


def get_url_artical(url, language='en'):
    try:
        url_i = Article(url="%s" % (url), language='en')
        url_i.download()
        url_i.parse()
        text = url_i.text
    except:
        text = ''
    return text


def get_sentiment_score(text, type='compound'):
    return SentimentIntensityAnalyzer().polarity_scores(text=text)[type]


def get_yf_news(ticker):
    return pd.DataFrame(news.get_yf_rss(ticker))['summary']

# %%
soup = finviz_parser_data("EH")
df = finviz_create_write_data(soup)

#%%
df['sentiment_score_headline'] = df.progress_apply(lambda x: get_sentiment_score(x['News Headline']), axis=1)

df = df[df['sentiment_score_headline'] != 0]

df['sentiment_score_details'] = df.progress_apply(lambda x: get_sentiment_score(get_url_artical(x['URL'])), axis=1)

#%%
news_list = get_yf_news('AAPL')

news_list.apply(lambda x: get_sentiment_score(x)).mean()

# %%
df.to_clipboard()