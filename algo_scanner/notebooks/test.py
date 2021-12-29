# %%
import re
import sqlite3
import time
from datetime import datetime

import finta
import multitasking
import pandas as pd
import yahoo_fin
import yahoo_finance
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pandas_datareader import data as pdr
from requests.api import head
from tqdm import tqdm
from yahoo_fin import news
from yahoo_fin import stock_info as si
from yahoofinancials import YahooFinancials

# %%
news_list = news.get_yf_rss('0700.HK')

#%%


#%%
# Import required module
import newspaper
  
# Assingn url
url = 'https://www.fool.com/earnings/call-transcripts/2021/12/02/ehang-holdings-limited-eh-q3-2021-earnings-call-tr/?source=eptyholnk0000202&utm_source=yahoo-host&utm_medium=feed&utm_campaign=article'
  
# Extract web data
url_i = newspaper.Article(url="%s" % (url), language='en')
url_i.download()
url_i.parse()

SentimentIntensityAnalyzer().polarity_scores(url_i.text)

#%%
# Import libraries
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen
from urllib.request import Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

n = 3

tickers = ['AAPL', 'TSLA']

# Get Data
finviz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
    resp = urlopen(req)
    html = BeautifulSoup(resp, features="lxml")
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

try:
    for ticker in tickers:
        df = news_tables[ticker]
        df_tr = df.findAll('tr')

        print('\n')
        print('Recent News Headlines for {}: '.format(ticker))

        for i, table_row in enumerate(df_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            td_text = td_text.strip()
            print(a_text, '(', td_text, ')')
            if i == n-1:
                break
except KeyError:
    pass

#%%
news_list[0]['summary_detail']['value']


#%%
analyzer = SentimentIntensityAnalyzer()


#%%
analyzer.polarity_scores('i like it!')


#%%
si.get_live_price('DIS')

#%%
df_ohlc = si.get_data('TSLA')

#%%
pd.concat([df_ohlc, finta.TA.VZO(df_ohlc)], axis=1).to_clipboard()


#%%
pd.concat([df_ohlc, finta.TA.EV_MACD(df_ohlc), finta.TA.MOM(df_ohlc), finta.TA.ROC(df_ohlc)], axis=1).to_clipboard()

#%%
df = pd.read_parquet(r'D:\Shiu\Documents\GitHub\kedro_equity\algo_scanner\data\01_raw\nasdaq_daliy_price.parquet')

# %%
df.sort_values(by=['ticker', 'date'], inplace=True)
# %%
df[df['ticker'] == 'AMD']

#%%
si.get_day_gainers(200)

#%%
si.get_company_info('EH')

#%%
si.get_company_info('TSLA')

#%%


# %%
info = yf.Ticker('EH').get_info()

# %%

