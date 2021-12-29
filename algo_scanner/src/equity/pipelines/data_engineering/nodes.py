from typing import Any, Dict

import multitasking
import pandas as pd
import yfinance as yf
from finta import TA
from pandas_datareader import data as pdr
from tqdm import tqdm
from yahoo_fin import stock_info as si

multitasking.set_max_threads('16')


def get_nasdaq_symbols() -> pd.DataFrame:
    """
    Function to obtain the nasdaq ticker symbols

    Returns:
        pd.DataFrame: the dataframe of the nasdaq symbols
    """
    return pdr.get_nasdaq_symbols()


def get_nasdaq_daily_price(df_tickers: pd.DataFrame) -> pd.DataFrame:
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

    return pd.concat(list_ohlc).dropna(how='all')


def get_nasdaq_company_info(df_tickers: pd.DataFrame,
                            df_info_loaded: pd.DataFrame) -> pd.DataFrame:

    if df_info_loaded is not None:
        if ~df_info_loaded.empty:
            df_tickers = df_tickers[~df_tickers.index.isin(df_info_loaded['symbol'])]
            print(df_tickers)

    list_info = []

    col = ['symbol',
           'quoteType',
           'shortName',
           'longBusinessSummary',
           'website',
           'exchange',
           'country',
           'financialCurrency',
           'sector',
           'industry',
           'recommendationMean',
           'isEsgPopulated',
           'fundFamily',
           'sectorWeightings',
           'holdings',
           'bondHoldings',
           'bondRatings',
           'equityHoldings',
           'stockPosition',
           'shortPercentOfFloat',
           'heldPercentInstitutions',
           'heldPercentInsiders',
           ]

    @multitasking.task
    def append_info(ticker):
        try:
            df = (pd
                  .DataFrame(yf.Ticker(ticker)
                             .get_info()
                             .items())
                  .set_index(0)
                  .T)
            df = df[[c for c in df.columns if c in col]]
            list_info.append(df)
        except:
            list_info.append(None)

    for i in df_tickers.index.to_list():
        print(i)
        append_info(i)

    with tqdm(total=len(df_tickers)) as pbar:
        while len(list_info) < len(df_tickers):
            cur_perc = len(list_info)
            pbar.update(cur_perc - pbar.n)
            if len(list_info) >= len(df_tickers):
                break

    return pd.concat(list_info + [df_info_loaded]).dropna(how='all')
