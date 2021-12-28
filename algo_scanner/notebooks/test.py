# %%
import re
import sqlite3
from datetime import datetime

import pandas as pd
import yahoo_fin
import yahoo_finance
import yfinance as yf
from pandas_datareader import data as pdr
from yahoo_fin import stock_info as si
from yahoofinancials import YahooFinancials

import multitasking
from tqdm import tqdm
import time
#%%
tickers = si.tickers_nasdaq()


#%%
list_ohlc = []


@multitasking.task
def append_ohlc(ticker):
    try:
        list_ohlc.append(si.get_data(ticker,
                                     index_as_date=False,
                                     start_date=None))
    except:
        list_ohlc.append(None)


for i in tickers:
    append_ohlc(i)
    
# %%


with tqdm(total=100) as pbar:
    while len(list_ohlc)/len(tickers) < 1:
        cur_perc = len(list_ohlc)/len(tickers) * 100
        pbar.update(cur_perc - pbar.n)
        if cur_perc == 100:
            break
#%%

while len(list_ohlc)/len(tickers) < 1:
    print(len(list_ohlc)/len(tickers))
    time.sleep(1)

# %%
df_ohlc = pd.concat(list_ohlc)

#%%
df_ohlc[df_ohlc['ticker'] == 'EH']
