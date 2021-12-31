
import multitasking
import finta
import pandas as pd
from tqdm import tqdm

list_collection = []


@multitasking.task
def list_collection_append(ticker):
    df_ohlc_one = df_ohlc[df_ohlc['ticker'] == ticker]
    df_ohlc_one = pd.concat([df_ohlc_one, finta.TA.EV_MACD(df_ohlc_one), finta.TA.ROC(df_ohlc_one), finta.TA.MOM(df_ohlc_one)], axis=1)
    list_collection.append(df_ohlc_one)


tickers = df_ohlc['ticker'].unique()
tickers = tickers[:20]

for i in tickers:
    print(i)
    list_collection_append(i)


with tqdm(total=len(tickers)) as pbar:
    while len(list_collection) < len(tickers):
        cur_perc = len(list_collection)
        pbar.update(cur_perc - pbar.n)
        if len(list_collection) >= len(tickers):
            break
