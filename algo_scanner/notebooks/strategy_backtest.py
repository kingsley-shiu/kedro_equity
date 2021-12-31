# %%
# SETUP SESSION
import pandas as pd
from os import add_dll_directory
from pathlib import Path
from dynaconf.default_settings import get

from kedro.framework.cli.utils import _add_src_to_path
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from kedro.framework.session.session import _activate_session
from kedro.framework.startup import _get_project_metadata

current_dir = Path.cwd()
project_path = current_dir.parent
metadata = _get_project_metadata(project_path)
_add_src_to_path(metadata.source_dir, project_path)
configure_project(metadata.package_name)
session = KedroSession.create(metadata.package_name, project_path)
_activate_session(session)
context = session.load_context()

#%%
def get_stock_price_hist(ticker, start_date='2020-01-01'):
    df_ohlc = context.io.load('nasdaq_daily_price')
    df_ohlc = df_ohlc.loc[df_ohlc['date'] >= start_date]
    df_ohlc = df_ohlc.loc[df_ohlc['ticker'] == ticker]
    df_ohlc.columns = df_ohlc.columns.str.capitalize()
    df_ohlc = df_ohlc.set_index('Date')
    return df_ohlc


#%%

from finta import TA as fta
from functools import reduce

# Buy hyperspace params:
buy_params = {
    "base_nb_candles_buy": 14,
    "ewo_high": 2.327,
    "ewo_low": -19.988,
    "low_offset": 0.975,
    "rsi_buy": 69
}

# Sell hyperspace params:
sell_params = {
    "base_nb_candles_sell": 24,
    "high_offset": 0.991,
    "high_offset_2": 0.997
}

def populate_indicators(dataframe: pd.DataFrame) -> pd.DataFrame:

    # Calculate all ma_buy values
    val = 14
    dataframe[f'ma_buy_{val}']= fta.EMA(dataframe)

    val= 24
    dataframe[f'ma_sell_{val}'] = fta.EMA(dataframe)
    
    dataframe['hma_50'] = fta.HMA(dataframe, 50)
    
    def EWO(dataframe, ema_length=5, ema2_length=35):
        df = dataframe.copy()
        ema1 = fta.EMA(df, ema_length)
        ema2 = fta.EMA(df, ema2_length)
        emadif = (ema1 - ema2) / df['Close'] * 100
        return emadif

    
    dataframe['sma_9'] = fta.SMA(ohlc, 9)
        # Elliot
    fast_ewo = 50
    slow_ewo = 200
    dataframe['EWO'] = EWO(dataframe, fast_ewo, slow_ewo)
        
        # RSI
    dataframe['rsi'] = fta.RSI(dataframe, 14)
    dataframe['rsi_fast'] = fta.RSI(dataframe, 4)
    dataframe['rsi_slow'] = fta.RSI(dataframe, 20)
    dataframe[['macd', 'signal']] = fta.EV_MACD(dataframe)


def populate_buy_trend(dataframe: pd.DataFrame) -> pd.DataFrame:

    conditions = []

    conditions.append(
        (
            dataframe['macd'] > dataframe['signal']
        )
    )

    conditions.append(
        (
            (dataframe['rsi_fast'] < 35) &
            (dataframe['Close'] < (dataframe[f'ma_buy_14'] * buy_params['low_offset'])) &
            (dataframe['EWO'] < buy_params['ewo_low']) &
            (dataframe['Volume'] > 0) &
            (dataframe['Close'] < (dataframe[f'ma_sell_24'] * sell_params['high_offset']))

        )
    )

    if conditions:
        dataframe.loc[
            reduce(lambda x, y: x | y, conditions),
            'buy'
        ]=1

    # return dataframe


def populate_sell_trend(dataframe: pd.DataFrame) -> pd.DataFrame:
    conditions = []

    conditions.append(
        (
            dataframe['macd'] <= dataframe['signal']
        )
    )

    if conditions:
        dataframe.loc[
            reduce(lambda x, y: x | y, conditions),
            'sell'
        ]=1

    # return dataframe


#%%

ohlc = get_stock_price_hist('TSLA')
# populate_indicators(ohlc)
# populate_buy_trend(ohlc)
# populate_sell_trend(ohlc)


# pd.concat([ohlc, fta.EVWMA(ohlc), fta.SQZMI(ohlc), fta.IFT_RSI(ohlc)], axis=1).to_clipboard()
#%%