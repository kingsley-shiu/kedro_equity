from typing import Any, Dict

import multitasking
import pandas as pd
import yfinance as yf
from finta import TA
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


def get_nasdaq_daily_price(df_tickers: pd.DataFrame) -> pd.DataFrame:
    list_ohlc = []

    @multitasking.task
    def append_ohlc(ticker):
        try:
            list_ohlc.append(si.get_data(ticker,
                                         index_as_date=False,
                                         start_date='2018-01-01')
                             )
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
        append_info(i)

    with tqdm(total=len(df_tickers)) as pbar:
        while len(list_info) < len(df_tickers):
            cur_perc = len(list_info)
            pbar.update(cur_perc - pbar.n)
            if len(list_info) >= len(df_tickers):
                break

    return pd.concat(list_info + [df_info_loaded]).dropna(how='all')


def get_nasdaq_share_held(df_tickers: pd.DataFrame,
                          df_share_held_loaded: pd.DataFrame) -> pd.DataFrame:

    if df_share_held_loaded is not None:
        if ~df_share_held_loaded.empty:
            df_tickers = df_tickers[~df_tickers.index.isin(df_share_held_loaded['ticker'])]

    list_holding = []

    tgt_return = ['% of Shares Held by All Insider',
                  '% of Shares Held by Institutions',
                  '% of Float Held by Institutions',
                  'Number of Institutions Holding Shares']

    @multitasking.task
    def append_holding(ticker):
        try:
            df_share_held = si.get_holders(ticker)['Major Holders'].dropna()
            if (not df_share_held.empty) & (df_share_held[1].to_list() == tgt_return):
                df_share_held = df_share_held.set_index(1).T
                df_share_held['ticker'] = ticker
                list_holding.append(df_share_held)
            else:
                list_holding.append(None)
        except:
            list_holding.append(None)

    for i in df_tickers.index.to_list():
        append_holding(i)

    with tqdm(total=len(df_tickers)) as pbar:
        while len(list_holding) < len(df_tickers):
            cur_perc = len(list_holding)
            pbar.update(cur_perc - pbar.n)
            if len(list_holding) >= len(df_tickers):
                break

    return pd.concat(list_holding).dropna(how='all')
