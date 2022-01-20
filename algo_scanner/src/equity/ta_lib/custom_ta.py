# %%
import numpy as np
import pandas as pd
import pandas_ta
import scipy
import talib
from scipy import signal

# %%


def hist_max(series, periods=250, fillna=False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).max()


def hist_min(series, periods=250, fillna=False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).min()


def savgol_smoother(series, window_length=15, polyorder=5, **kwargs):
    if len(series) >= window_length:
        _series_smooth = pd.Series(signal.savgol_filter(x=series, window_length=window_length, polyorder=polyorder, **kwargs))
    else:
        _series_smooth = pd.Series([np.nan for x in series])
    _series_smooth.index = series.index
    return _series_smooth


def local_max(series, order=5, **kwargs):
    series_index_reset = series.reset_index()

    col = series_index_reset.columns[-1]

    series_index_reset['local_max'] = series_index_reset.index.isin(signal.argrelextrema(np.array(series_index_reset[col]), comparator=np.greater, order=order, **kwargs)[0])
    series_index_reset['local_max'] = series_index_reset['local_max'].map({True: 1, False: 0})
    series_index_reset.set_index('index', inplace=True)
    series_index_reset.index.name = None

    return series_index_reset['local_max']


def local_min(series, order=5, **kwargs):
    series_index_reset = series.reset_index()

    col = series_index_reset.columns[-1]

    series_index_reset['local_min'] = series_index_reset.index.isin(signal.argrelextrema(np.array(series_index_reset[col]), comparator=np.less, order=order, **kwargs)[0])
    series_index_reset['local_min'] = series_index_reset['local_min'].map({True: 1, False: 0})
    series_index_reset.set_index('index', inplace=True)
    series_index_reset.index.name = None

    return series_index_reset['local_min']


def gradient_trend(series, edge_order=1, window_period=20, up=True, **kwargs):
    '''for continous no. of days for increasing'''
    _dict_TF = {True: 1, False: 0}
    try:
        if up:
            _gradient_series = pd.Series(np.gradient(series, edge_order) >= 0)
        else:
            _gradient_series = pd.Series(np.gradient(series, edge_order) <= 0)

        _gradient_series = _gradient_series.map(_dict_TF)
        _gradient_series_rolling_sum = _gradient_series.rolling(window=window_period).sum()
        _gradient_series_win_trend = (_gradient_series_rolling_sum >= window_period).map(_dict_TF).astype(int)
    except:
        _gradient_series_win_trend = pd.Series([np.nan for x in series])

    _gradient_series_win_trend.index = series.index
    return _gradient_series_win_trend


def gradient(series, edge_order=1):
    _dict_TF = {True: 1, False: 0}
    try:
        _gradient_series = pd.Series(np.gradient(series, edge_order))
    except:
        _gradient_series = pd.Series([np.nan for x in series])
    _gradient_series.index = series.index
    return _gradient_series


def zz_indicator(high, low, depth=10, deviation=5):
    zz = {high.index.min(): 0}

    def zz_update_high(zz, curr_pos, high, low, depth, deviation):
        # condition 1 max from previous min/max to curr_pos + depth
        check_range = [list(zz.keys())[-1],
                       min(curr_pos+depth, high.index.max())+1]

        if curr_pos == high.loc[check_range[0]:check_range[1]].idxmax():
            if zz[check_range[0]] != -1:  # if not low, must be high or init
                # check if ths high is higher then previous, if so remove the previous one and update this one
                if high[curr_pos] > high[check_range[0]]:
                    zz.popitem()
                    zz.update({curr_pos: 1})
            elif (zz[check_range[0]] != 1):
                # if previous is low, check the deviation is met and update this one. else ignore this high.
                # only update the zz when this is new indicator
                if (high[curr_pos] - low[check_range[0]] >= low[check_range[0]] * deviation/100) & ~(curr_pos in zz):
                    zz.update({curr_pos: 1})

    def zz_check_low(zz, curr_pos, high, low, depth, deviation):
        check_range = [list(zz.keys())[-1],
                       min(curr_pos+depth, high.index.max())+1]
        if curr_pos == low.loc[check_range[0]:check_range[1]].idxmin():
            if zz[check_range[0]] != 1:  # if not high, must be low or init
                # check if ths low is lower then previous, if so remove the previous one and update this one
                if low[curr_pos] < low[check_range[0]]:
                    zz.popitem()
                    zz.update({curr_pos: -1})
            elif zz[check_range[0]] != -1:
                # if previous is high, check the deviation is met and update this one. else ignore this low.
                # only update the zz when this is new indicator
                if (high[check_range[0]] - low[curr_pos] >= high[check_range[0]] * deviation / 100) & ~(curr_pos in zz):
                    zz.update({curr_pos: -1})

    for i in range(high.index.min() + depth, high.index.max()):
        zz_check_low(zz, curr_pos=i, high=high, low=low, depth=depth, deviation=deviation)
        zz_update_high(zz, curr_pos=i, high=high, low=low, depth=depth, deviation=deviation)
    return pd.Series(high.index.map(zz).fillna(0).astype(int), high.index)


def zz_indicator_single(close, depth=10, deviation=5):
    zz = {close.index.min(): 0}

    def zz_update_high(zz, curr_pos, close, depth, deviation):
        # condition 1 max from previous min/max to curr_pos + depth
        check_range = [list(zz.keys())[-1],
                       min(curr_pos+depth, close.index.max())+1]

        if curr_pos == close.loc[check_range[0]:check_range[1]].idxmax():
            if zz[check_range[0]] != -1:  # if not low, must be high or init
                # check if ths high is higher then previous, if so remove the previous one and update this one
                if close[curr_pos] > close[check_range[0]]:
                    zz.popitem()
                    zz.update({curr_pos: 1})
            elif (zz[check_range[0]] != 1):
                # if previous is low, check the deviation is met and update this one. else ignore this high.
                # only update the zz when this is new indicator
                if (close[curr_pos] - close[check_range[0]] >= close[check_range[0]] * deviation/100) & ~(curr_pos in zz):
                    zz.update({curr_pos: 1})

    def zz_check_low(zz, curr_pos, close, depth, deviation):
        check_range = [list(zz.keys())[-1],
                       min(curr_pos+depth, close.index.max())+1]
        if curr_pos == close.loc[check_range[0]:check_range[1]].idxmin():
            if zz[check_range[0]] != 1:  # if not high, must be low or init
                # check if ths low is lower then previous, if so remove the previous one and update this one
                if close[curr_pos] < close[check_range[0]]:
                    zz.popitem()
                    zz.update({curr_pos: -1})
            elif zz[check_range[0]] != -1:
                # if previous is high, check the deviation is met and update this one. else ignore this low.
                # only update the zz when this is new indicator
                if (close[check_range[0]] - close[curr_pos] >= close[check_range[0]] * deviation / 100) & ~(curr_pos in zz):
                    zz.update({curr_pos: -1})

    for i in range(close.index.min() + depth, close.index.max()):
        zz_check_low(zz, curr_pos=i, close=close, depth=depth, deviation=deviation)
        zz_update_high(zz, curr_pos=i, close=close, depth=depth, deviation=deviation)
    return pd.Series(close.index.map(zz).fillna(0).astype(int), close.index)


def zz_contract_expand_pct(high, low, zz_ind):
    s_high_low = pd.Series(np.where(zz_ind == 1, high, (np.where(zz_ind == -1, low, None))), index=zz_ind.index).dropna()
    s_contract_expand_pct = np.diff(s_high_low)/s_high_low.shift(1)[1:]
    return s_contract_expand_pct


def zz_contract_expand(high, low, zz_ind):
    s_high_low = pd.Series(np.where(zz_ind == 1, high, (np.where(zz_ind == -1, low, None))), index=zz_ind.index).dropna()
    s_contract_expand = pd.Series(np.diff(s_high_low), index=s_high_low[1:].index)
    return s_contract_expand


def zz_contraction_group(zz, zz_val, zz_pct, by_pct=False):
    if by_pct:
        if (zz == -1).sum() > (zz == 1).sum():
            s_gp_start = pd.Series((np.diff(zz_pct[zz == -1]) < 0)[1:], index=zz_pct[zz == 1].index[1:]).cumsum()
        else:
            s_gp_start = pd.Series((np.diff(zz_pct[zz == -1]) < 0), index=zz_pct[zz == 1].index[1:len(zz_pct[zz == -1])]).cumsum()
        s_gp_end = pd.Series(np.diff(zz_pct[zz == -1]) < 0, index=zz_pct[zz == -1].index[1:]).cumsum()
    else:
        if (zz == -1).sum() > (zz == 1).sum():
            s_gp_start = pd.Series((np.diff(zz_val[zz == -1]) < 0)[1:], index=zz_val[zz == 1].index[1:len(zz_val[zz == -1])]).cumsum()
        else:
            s_gp_start = pd.Series((np.diff(zz_val[zz == -1]) < 0), index=zz_val[zz == 1].index[1:len(zz_val[zz == -1])]).cumsum()
        s_gp_end = pd.Series(np.diff(zz_val[zz == -1]) < 0, index=zz_val[zz == -1].index[1:]).cumsum()
    df = pd.DataFrame(zz, columns=['zz'])
    df['gp_start'] = s_gp_start
    df['gp_end'] = s_gp_end
    s_gp = df['gp_start'].fillna(df['gp_end']).ffill().fillna(0)
    return s_gp


def chip_vweap(high, low, close, volume, window=50):
    typical_price = pandas_ta.hlc3(high, low, close)
    adj_vol = (close - (high + low)/2).pow(2) * volume
    return talib.EMA(typical_price * adj_vol, window) / talib.EMA(adj_vol, window)


def p_strength(close, ref_trend, window=200, rolling_trend=True):
    '''ref_trend is chip'''
    if rolling_trend:
        rolling = ref_trend.rolling(window)
        zScore = (close - rolling.mean()) / rolling.std()
    else:
        rolling = ref_trend
        zScore = (close - rolling) / rolling.std()
    return scipy.stats.norm.cdf(zScore)


def ifisher(series):
    return (np.exp(series * 2) - 1)/(np.exp(series * 2) + 1)


def rs_price(series, benchmark):
    series, benchmark = series.align(benchmark, join='left')
    benchmark.ffill(inplace=True)
    rs_p = (series.divide(benchmark))
    return rs_p


def rs_ratio(series, benchmark, periods, smooth=True, window_length=31, polyorder=3):

    # def rs_ratio(prices_df, benchmark, window=10):
    # from numpy import mean, std
    # for series in prices_df:
    #     rs = (prices_df[series].divide(benchmark)) * 100
    #     rs_ratio = rs.rolling(window).mean()
    #     rel_ratio = 100 + ((rs_ratio - rs_ratio.mean()) / rs_ratio.std() + 1)
    #    prices_df[series] = rel_ratio
    # prices_df.dropna(axis=0, how='all', inplace=True)
    # return prices_df

    # rs = (series.divide(benchmark)) * 100
    # rs_ratio = rs.rolling(periods).mean()
    # rel_ratio = ((rs_ratio - rs_ratio.mean()) / rs_ratio.std() + 1)

    rs = (series.divide(benchmark))
    rs_ratio = rs.rolling(periods).mean()
    rel_ratio = (rs - rs_ratio)/rs.rolling(periods).std()
    # rel_ratio = (rs_ratio - rs_ratio.rolling(periods).mean())/rs_ratio.rolling(periods).std()

    if smooth:
        # ! moving function will repaint
        rel_ratio = savgol_smoother(rel_ratio.dropna(), window_length=window_length, polyorder=polyorder)
        # rel_ratio = pd.Series(triple_exponential_smoothing(rel_ratio.dropna().values, 12, 0.1, 0.1, 0.1, 0.5),
        #                       index=rel_ratio.dropna().index)
    else:
        rel_ratio = rel_ratio.dropna()
    return rel_ratio


def rs_momenmtum(rs_r, smooth=True, window_length=31, polyorder=3):
    # s = rs_r.diff()
    # rs_m = 100 + ((s - s.rolling(periods).mean())/s.rolling(periods).std() + 1)

    if smooth:
        # ! moving function will repaint
        rs_m = savgol_smoother(pandas_ta.slope(rs_r).dropna(), window_length=window_length, polyorder=polyorder)
        # rs_m = pd.Series(triple_exponential_smoothing(pandas_ta.slope(rs_r).dropna().values, 12, 0.1, 0.1, 0.1, 0.5),
        #                  index=pandas_ta.slope(rs_r).dropna().index)
    else:
        rs_m = pandas_ta.slope(rs_r).dropna()

    return rs_m


def moving_avarage_smoothing(X, k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S


def exponential_smoothing(X, α):
    S = np.zeros(X.shape[0])
    S[0] = X[0]
    for t in range(1, X.shape[0]):
        S[t] = α * X[t-1] + (1 - α) * S[t-1]
    return S


def double_exponential_smoothing(X, α, β):
    '''https://github.com/srv96/Data-Analytics-with-python/blob/master/TimeSeriesSmoothingTechiniques/smoothing_techiniques.py'''
    S, A, B = (np.zeros(X.shape[0]) for i in range(3))
    S[0] = X[0]
    for t in range(1, X.shape[0]):
        A[t] = α * X[t] + (1 - α) * S[t-1]
        B[t] = β * (A[t] - A[t-1]) + (1 - β) * B[t-1]
        S[t] = A[t] + B[t]
    return S


def triple_exponential_smoothing(X, L=15, α=0.28, β=0, γ=0, ϕ=1):
    '''triple_exponential_smoothing(time_series,12,0.1,0.1,0.1,0.5)'''
    def sig_ϕ(ϕ, m):
        return np.sum(np.array([np.power(ϕ, i) for i in range(m+1)]))

    C, S, B, F = (np.zeros(X.shape[0]) for i in range(4))
    S[0], F[0] = X[0], X[0]
    B[0] = np.mean(X[L:2*L] - X[:L]) / L
    m = 12
    sig_ϕ = sig_ϕ(ϕ, m)
    for t in range(1, X.shape[0]):
        S[t] = α * (X[t] - C[t % L]) + (1 - α) * (S[t-1] + ϕ * B[t-1])
        B[t] = β * (S[t] - S[t-1]) + (1-β) * ϕ * B[t-1]
        C[t % L] = γ * (X[t] - S[t]) + (1 - γ) * C[t % L]
        F[t] = S[t] + sig_ϕ * B[t] + C[t % L]
    return S


def sctr(close, bound=False):
    SCTR = pd.Series(None, index=close.index)
    Fc = 1
    LT_Rate = 60
    MT_Rate = 30
    ST_Rate = 10

    # Long term input
    LT_EMA = 200
    LT_ROC = 125

    # Middle term input
    MT_EMA = 50
    MT_ROC = 20

    # Short term input
    ST_RSI = 14
    ST_EMA = 9

    if len(close) >= LT_EMA:
        # Long term integer
        LT_EMAV = (close / pandas_ta.ema(close, LT_EMA)) * 100
        LT_ROCV = pandas_ta.roc(close, LT_ROC)

        # Middle term integer
        MT_EMAV = (close / pandas_ta.ema(close, MT_EMA)) * 100
        MT_ROCV = pandas_ta.roc(close, MT_ROC)

        # Short term integer
        ST_RSIV = pandas_ta.rsi(close, ST_RSI)
        ST_PPO_SLOPE = (pandas_ta.slope(pandas_ta.ppo(close)['PPOh_12_26_9'], 3) + 1) * 50
        # ST_EMAV = (close / pandas_ta.ema(close, ST_EMA)) * 100

        LT_Val = LT_Rate * 0.01 * ((LT_EMAV + LT_ROCV)/2)
        MT_Val = MT_Rate * 0.01 * ((MT_EMAV + MT_ROCV)/2)
        ST_Val = ST_Rate * 0.01 * ((ST_RSIV + ST_PPO_SLOPE)/2)
        # ST_Val = ST_Rate * 0.01 * ((ST_EMAV + ST_RSIV)/2)

        # ST_Val = (0.01 * ST_Rate * ST_RSIV)

        if bound:
            # SCTR = pd.Series(np.where(SCTR >= 99.9, 99.9, np.where(SCTR <= 0, 0, SCTR)), close.index)
            LT_Val = pd.Series(np.where(LT_Val >= 60, 60, np.where(LT_Val.isnull(), None, LT_Val)), close.index)
            MT_Val = pd.Series(np.where(MT_Val >= 30, 30, np.where(MT_Val.isnull(), None, MT_Val)), close.index)
            ST_Val = pd.Series(np.where(ST_Val >= 10, 10, np.where(ST_Val.isnull(), None, ST_Val)), close.index)

        SCTR = (Fc * (LT_Val + MT_Val + ST_Val))
        SCTR.name = 'SCTR'
        SCTR = SCTR.astype(float)
    return SCTR


def ibd(close):
    ibd_score = (close/pandas_ta.sma(close, 63) * 0.4 +
                 close/pandas_ta.sma(close, 126) * 0.2 +
                 close/pandas_ta.sma(close, 189) * 0.2 +
                 close/pandas_ta.sma(close, 252) * 0.2)
    return ibd_score

# %%
