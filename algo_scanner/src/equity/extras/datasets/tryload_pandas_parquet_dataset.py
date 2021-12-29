from typing import Any, Dict
import pandas as pd

from kedro.extras.datasets.pandas import ParquetDataSet
from kedro.io.core import Version


class tryload_pandas_parquet(ParquetDataSet):
    
    def __init__(self, filepath: str, load_args: Dict[str, Any] = None, save_args: Dict[str, Any] = None, version: Version = None, credentials: Dict[str, Any] = None, fs_args: Dict[str, Any] = None) -> None:
        super().__init__(filepath, load_args=load_args, save_args=save_args, version=version, credentials=credentials, fs_args=fs_args)
        
    def _load(self) -> pd.DataFrame:
        try:
            return super()._load()
        except:
            return None
