# %%
# SETUP SESSION
from os import add_dll_directory
from pathlib import Path

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
df = context.io.load('nasdaq_company_info')
df.dropna(how='all')


# %%
df.to_clipboard()
# %%
import yfinance as yf
import pandas as pd
ticker = 'IOO'
df = (pd
      .DataFrame(yf.Ticker(ticker)
                 .get_info()
                 .items())
      .set_index(0)
      .T)

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
       ]

df = df[[c for c in df.columns if c in col]]

