# %%
import re
import sqlite3
from datetime import datetime

import pandas as pd
from requests.api import head
import yahoo_fin
import yahoo_finance
import yfinance as yf
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from yahoofinancials import YahooFinancials
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import multitasking
from tqdm import tqdm
import time

import finta

from yahoo_fin import news

# %%
news_list = news.get_yf_rss('AAPL')

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

