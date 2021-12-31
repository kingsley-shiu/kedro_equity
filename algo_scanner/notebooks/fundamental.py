#%%
import pandas as pd
from pandas.io.formats.format import DataFrameFormatter
import yahoofinancials
import yfinance as yf
from yahoo_fin import stock_info as si
from yahoo_fin import options as opt

#%%
si.get_analysts_info('TSLA').keys()

#%%
bal1 = si.get_balance_sheet('TSLA', yearly=False).T
bal2 = si.get_balance_sheet('AAPL', yearly=False).T

#%%
si.get_financials('AAPL')['quarterly_income_statement']
# %%


#%%
si.get_analysts_info('O')['EPS Revisions']


#%%
si.get_futures()

#%%

list_holding = []

#%%
ticker = 'AAPL'
si.get_holders(ticker)['Major Holders']

#%%
exp_date = opt.get_expiration_dates('AAPL')
opt.get_calls('AAPL', [0])

#%%
# %%
opt.get_puts('TSLA')
#%%
