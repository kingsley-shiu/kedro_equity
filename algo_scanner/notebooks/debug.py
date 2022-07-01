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
# %%
df_share = context.io.load('nasdaq_daily_price')

#%%
df_share[df_share['ticker'] == 'TSLA']
#%%
df_share.sort_values(by='Number of Institutions Holding Shares').to_clipboard()

#%%
from yahoo_fin import stock_info as si

list_holding = []


def append_holding(ticker):
    df_share_held = si.get_holders(ticker)['Major Holders'].dropna()
    if not df_share_held.empty:
        df_share_held = df_share_held.set_index(1).T
        df_share_held['ticker'] = ticker
        list_holding.append(df_share_held)
    else:
        list_holding.append(None)

# %%
import multitasking
from tqdm import tqdm

list_holding = []
df_tickers = df_tickers.head(10)

tgt_return = ['% of Shares Held by All Insider',
              '% of Shares Held by Institutions',
              '% of Float Held by Institutions',
              'Number of Institutions Holding Shares']

@multitasking.task
def append_holding(ticker):
    try:
        df_share_held = si.get_holders(ticker)['Major Holders'].dropna()
        if not df_share_held.empty & (df_share_held[1].to_list() == tgt_return):
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

#%%
list_holding[5]

#%%

si.get_holders('TSLA')['Major Holders'][1].to_list() == ['Previous Close',
                                                        'Open',
                                                        'Bid',
                                                        'Ask',
                                                        "Day's Range",
                                                        '52 Week Range',
                                                        'Volume',
                                                        'Avg. Volume']
