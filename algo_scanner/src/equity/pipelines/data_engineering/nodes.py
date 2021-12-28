from typing import Any, Dict

import multitasking
import pandas as pd
from pandas_datareader import data as pdr
from tqdm import tqdm
from yahoo_fin import stock_info as si


def get_nasdaq_symbols() -> pd.DataFrame:
    """
    Function to obtain the nasdaq ticker symbols

    Returns:
        pd.DataFrame: the dataframe of the nasdaq symbols
    """
    return pdr.get_nasdaq_symbols()


def get_nasdaq_daily_price(df_tickers) -> pd.DataFrame:
    list_ohlc = []

    @multitasking.task
    def append_ohlc(ticker):
        try:
            list_ohlc.append(si.get_data(ticker,
                                         index_as_date=False,
                                         start_date=None))
        except:
            list_ohlc.append(None)

    for i in df_tickers.index.to_list():
        append_ohlc(i)

    with tqdm(total=len(df_tickers)) as pbar:
        while len(list_ohlc) < len(df_tickers):
            cur_perc = len(list_ohlc)
            pbar.update(cur_perc - pbar.n)
            if len(list_ohlc) >= len(df_tickers):
                break

    return pd.concat(list_ohlc)
