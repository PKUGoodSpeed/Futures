## Basic
import os
import time
import numpy as np
import pandas as pd
## Using ginkgo database
import gsampler as gs
from light.util import selectDates
## Using multi threading 
from multithrds import MyPool
## H5 io
from h5io import write_to_h5

def _get_raw_data(symbol, colo, date, n_levels=5, start_time="09:05:00", end_time="13:25:00"):
    '''
    Sampling raw market data.
    What we will get:
        ask/bid prices
        ask/bid book sizes
        trade information:
            trade prices
            trade sides
            trade volumes
    Other features may be added in later
    '''
    assert n_levels > 0 and n_levels <= 5, "n_levels should be between 1 and 5."
    features = []
    for i in range(n_levels):
        features.append(gs.FeatureAskPrice(symbol=symbol, level=i))
        features.append(gs.FeatureAskQty(symbol=symbol, level=i))
    for i in range(n_levels):
        features.append(gs.FeatureBidPrice(symbol=symbol, level=i))
        features.append(gs.FeatureBidQty(symbol=symbol, level=i))
    features.append(gs.FeatureTradePrice(symbol = symbol))
    features.append(gs.FeatureTradeSide(symbol = symbol))
    features.append(gs.FeatureTradeSize(symbol = symbol))
    features.append(gs.FeatureTickerIdCast())
    sampler = gs.SamplerRunner(features = features, 
                                dates = [date],
                                colo = colo, 
                                cb_version = 'v2')
    sampler.rerun()
    df = sampler.load_data_df()
    df = df.between_time(start_time=start_time, end_time=end_time)
    df = df.fillna(0.)
    df.columns = np.concatenate([
        np.concatenate([['ask_price_{0}'.format(i), 'ask_size_{0}'.format(i)] for i in range(n_levels)]),
        np.concatenate([['bid_price_{0}'.format(i), 'bid_size_{0}'.format(i)] for i in range(n_levels)]),
        ['trade_price', 'trade_side', 'trade_volume'],
        ['ticker_id', 'seq_num', 'time']
    ])
    df.drop('ticker_id', axis=1, inplace=True)
    df.drop('seq_num', axis=1, inplace=True)
    df.drop('time', axis=1, inplace=True)               ## So far we do not consider time
    return df


def _get_raw_data_wrapper(kargs):
    ''' Wrapper for multithreading '''
    return _get_raw_data(**kargs)

def sample_market_data(symbol, colo, dates, data_path='../data', n_threads=1, save='csv', **kargs):
    assert n_threads > 0, "number of threads should be positive."
    kargs_list = []
    dates = selectDates(dates)
    for date in dates:
        info = {
            'symbol': symbol,
            'colo': colo,
            'date': date
        }
        kargs_list.append(dict(info, **kargs))

    start_time = time.time()

    df_list = []
    if n_threads == 1:
        for kargs in kargs_list:
            df_list.append(_get_raw_data(**kargs))
    else:
        pool = MyPool(n_threads)
        df_list = pool.map(_get_raw_data_wrapper, kargs_list)
        pool.close()
        pool.join()

    sampling_time = time.time() - start_time

    if save is not None:
        if not os.path.exists(data_path):
            os.system('mkdir {0}'.format(data_path))
        if not os.path.exists('{0}/{1}'.format(data_path, colo)):
            os.system('mkdir {0}/{1}'.format(data_path, colo))
    
    start_time = time.time()
    
    if save == 'csv':
        print "Saving data into folder: {0}/{1}".format(data_path, colo)
        for df, date in zip(df_list, dates):
            df.to_csv(data_path+'/{0}/{1}-{2}.csv'.format(colo, symbol, date), index=False)
    elif save == 'h5':
        print "Saving data into folder: {0}/{1}".format(data_path, colo)
        for df, date in zip(df_list, dates):
            write_to_h5(data_path+'/{0}/{1}-{2}.h5'.format(colo, symbol, date), df)
    
    saving_time = time.time() - start_time

    print "Time Usage: "
    print "sampling: ", sampling_time, " s"
    print "saving: ", saving_time, " s"

    return df_list

def _test1():
    from IPython.display import display
    symbol = 'KRW_G0.KRX'
    colo = "krx"
    date = '20171129'
    df = _get_raw_data(symbol, colo, date, n_levels=5, start_time='09:03', end_time='15:30')
    print "The columns of the sampled dataframe:"
    print df.columns
    print "The top 3 rows of the sampled data:"
    display(df[:3])


def _test2():
    kargs = {
        'n_levels': 5,
        'start_time': "09:05:00",
        'end_time': "13:25:00"
    }
    symbol = 'KRW_G0.KRX'
    colo = "krx"
    dates = selectDates('20171101~20171117')
    sample_market_data(symbol, colo, dates, data_path='../data', n_threads=1, save='csv', **kargs)


def _test3():
    kargs = {
        'n_levels': 5,
        'start_time': "09:05:00",
        'end_time': "13:25:00"
    }
    symbol ='TX_G0.TWF'
    colo = 'twf'
    dates = "20171101~20171117"
    sample_market_data(symbol, colo, dates, data_path='../data', n_threads=8, save='h5', **kargs)


if __name__ == '__main__':
    #_test1()
    #_test2()
    _test3()
    
'''
Results:
    Sampling:
        1. Using 1 thread: 69.3473489285  s
        2. Using 8 threads: 10.1351988316  s
    Saving:
        1. Saving as csv: 64.0884530544  s
        2. Saving as h5: 3.17948508263  s
'''