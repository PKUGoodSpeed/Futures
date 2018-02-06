## basic
import os
import time
import numpy as np
import pandas as pd
## multi-threading
from multithrds import MyPool
## preprocessing market data
from preprocess import preprocess
## dataReader
from h5io import load_from_h5
## show image features
from plotter import showImageFeatures


def _get_labels(prev, post, n_levels, n_target_levels):
    """
    Extracting possible market changes due to other people"s behaviors
    prev: previous market snapshot
    post: next market snapshot
    n_levels: number of levels we actually sampled
    n_target_levels: Upto how many levels we consider the book changes
    return: target, array/list type with 4 * n_target_levels + 6 entries: [
        buy trade, sell trade,
        ask_price increase, ask_price decrease,
        ask_book_0 increase, ask_book_0 decrease ...
        bid_price increase, bid_price decrease,
        bid_book_0 increase, bid_book_0 decrease ...
    ]
    """
    assert(n_target_levels <= n_levels)
    target = np.zeros(4*n_target_levels + 6)
    if prev[4*n_levels + 1]:
        return None
    if post[4*n_levels + 1] == 1:
        target[0] = 1
    if post[4*n_levels + 1] == -1:
        target[1] = 1
    if prev[0] < post[0]:
        target[2] = 1
    elif prev[0] > post[0]:
        target[3] = 1
    else:
        for i in range(n_target_levels):
            if prev[2*i]*post[2*i] == 0:
                continue
            if prev[2*i+1] < post[2*i+1]:
                target[4+2*i] = 1
            elif prev[2*i+1] > post[2*i+1]:
                target[4+2*i+1] = 1
    if prev[2*n_levels] < post[2*n_levels]:
        target[4+2*n_target_levels] = 1
    elif prev[2*n_levels] > post[2*n_levels]:
        target[4+2*n_target_levels+1] = 1
    else:
        for i in range(n_target_levels):
            if prev[2*n_levels+2*i]*post[2*n_levels+2*i] == 0:
                continue
            if prev[2*n_levels+2*i+1] < post[2*n_levels+2*i+1]:
                target[6+2*n_target_levels+2*i] = 1
            elif prev[2*n_levels+2*i+1] > post[2*n_levels+2*i+1]:
                target[6+2*n_target_levels+2*i+1] = 1
    return target
def _get_labels_wrapper(kargs):
    return _get_labels(**kargs)


""" ============= Extracting our own features ============= """
def _get_image_features(vecs, n_levels, look_back, window):
    """
    vecs: a block of market snapshot
    n_levels: number of levels we sampled
    look_back: number of look back ticks
    window: the window of the image
    Getting look_backXwindow image feature with 4 channels:
        - ask book sizes
        - bid book sizes
        - aggressive buy trade
        - aggressive sell trade
    """
    assert len(vecs) == look_back
    img = np.zeros((look_back, window, 4))
    ask_base = int(vecs[look_back-1][0] - int((window-1)/2))
    bid_base = int(vecs[look_back-1][2*n_levels] - int(window/2))
    for i in range(look_back):
        for j in range(n_levels):
            level = vecs[i][2*j] - ask_base
            if level>=0 and level<window:
                img[i][level][0] = vecs[i][2*j + 1]
        for j in range(n_levels):
            level = vecs[i][2*n_levels + 2*j] - bid_base
            if level>=0 and level<window:
                img[i][level][1] = vecs[i][2*n_levels + 2*j + 1]
        if vecs[i][4*n_levels + 1] == 1:
            level = vecs[i][4*n_levels] - ask_base
            if level>=0 and level<window:
                img[i][level][2] = vecs[i][4*n_levels+2]
        if vecs[i][4*n_levels + 1] == -1:
            level = vecs[i][4*n_levels] - bid_base
            if level>=0 and level<window:
                img[i][level][3] = vecs[i][4*n_levels+2]
    return img
def _get_image_features_wrapper(kargs):
    return _get_image_features(**kargs)


def _get_sequence_feature(vecs, idx):
    """
    Getting time_series features
    """
    assert len(vecs)
    assert idx < len(vecs[0])
    assert vecs[:, idx]
def _get_sequence_feature_wrapper(kargs):
    return _get_sequence_feature(**kargs)


class GetFeatures:
    _mds = None
    _n_levels = None
    _n_target_levels = None
    _look_back = None
    _window = None
    _cutoff = None


    def __init__(self, symbol, colo, path="../data", data_format="h5", tick_size=1., 
    n_levels=5, n_target_levels=3, look_back=32, window=12, cutoff=500):
        self._mds = []
        print "Reading data from {0}...".format(path)
        start_time = time.time()
        data_path = path + "/" + colo
        assert os.path.exists(data_path), "Data path, {0}, does not exist.".format(data_path)
        for f in os.listdir(data_path):
            filename = data_path + "/" + f
            if symbol in f and f[-2:] == "h5" and data_format == "h5":
                df = load_from_h5(filename)
            elif symbol in f and f[-3:] == "csv" and data_format == "csv":
                df = pd.read_csv(filename)
            if df.empty:
                print "Data file: {0} is empty.".format(filename)
            else:
                self._mds.append(preprocess(df, tick_size))
        print "Reading data finished."
        print "Time Usage: ", time.time() - start_time, " s"
        self._n_levels = n_levels
        self._n_target_levels = n_target_levels
        self._look_back = look_back
        self._window = window
        self._cutoff = cutoff


    def _get_market_changes(self, mat, n_threads=1):
        print("Extracting market changes...")
        kargs_list = [{
            "prev": mat[i+self._look_back-1],
            "post": mat[i+self._look_back],
            "n_levels": self._n_levels,
            "n_target_levels": self._n_target_levels
        } for i in range(len(mat) - self._cutoff)]
        
        pool = MyPool(n_threads)
        targets = pool.map(_get_labels_wrapper, kargs_list)
        pool.close()
        pool.join()
        return targets


    def _get_image_features(self, mat, n_threads=1):
        print("Extracting image features...")
        kargs_list = [{
            "vecs": mat[i: i+self._look_back],
            "n_levels": self._n_levels,
            "look_back": self._look_back,
            "window": self._window
        } for i in range(len(mat) - self._cutoff)]
        
        pool = MyPool(n_threads)
        imgs = pool.map(_get_image_features_wrapper, kargs_list)
        pool.close()
        pool.join()
        return imgs


    def _get_ask_price_features(self, mat):
        print("Extracting ask price features...")
        ask_price_seq = mat[:, 0]
        return [ask_price_seq[i: i+self._look_back] for i in range(len(mat) - self._cutoff)]


    def _get_bid_price_features(self, mat):
        print("Extracting bid price features...")
        bid_price_seq = mat[:, 2*self._n_levels]
        return [bid_price_seq[i: i+self._look_back] for i in range(len(mat) - self._cutoff)]


    def get_features(self, n_threads=1):
        print("Generating training sets...")
        start_time = time.time()
        feature_df = pd.DataFrame({
            "target": [],
            "images": [],
            "ask_price_seq": [],
            "bid_price_seq":[]
        })
        for df in self._mds:
            mat = df.values
            print len(mat)
            feature_df = feature_df.append(pd.DataFrame({
                "target": self._get_market_changes(mat, n_threads),
                "images": self._get_image_features(mat, n_threads),
                "ask_price_seq": self._get_ask_price_features(mat),
                "bid_price_seq": self._get_bid_price_features(mat)
            }).dropna(), ignore_index=True)
        print("Job finished!")
        print("Time usage: " + str(time.time() - start_time) + " s")
        return feature_df


def _test1():
    from IPython.display import display
    df = load_from_h5("../data/twf/TX_G0.TWF-20171101.h5")
    print df.columns
    mat = df.values[:20]
    print mat
    df = preprocess(df,tick_size=1.)
    mat = df.values[:20]
    for i in range(19):
        print i, mat[i]
    for i in range(19):
        print i, _get_labels(mat[i], mat[i+1], 5, 3)


def _test2():
    from IPython.display import display
    df = load_from_h5("../data/twf/TX_G0.TWF-20171101.h5")
    df = preprocess(df,tick_size=1.)
    look_back = 10
    window = 10
    n_levels = 5
    img = _get_image_features(df.values[:look_back], n_levels, look_back, window)
    filename = "./test/sample.png"
    titles = ["ask_book_sizes", "bid_book_sizes", "buy_trades", "sell_trades"]
    showImageFeatures(img, titles, filename)


def _test3():
    feature_kargs = {
        "symbol": "TX_G0.TWF", 
        "colo": "twf",
        "path": "../data",
        "data_format": "h5",
        "tick_size": 1.,
        "n_levels": 5,
        "n_target_levels": 3,
        "look_back": 32,
        "window": 12,
        "cutoff": 500
    }
    A = GetFeatures(**feature_kargs)
    target = A._get_market_changes(A._mds[0].values, n_threads=8)
    print target[:20]
    imgs = A._get_image_features(A._mds[0].values, n_threads=8)
    filename = "./test/sample5.png"
    titles = ["ask_book_sizes", "bid_book_sizes", "buy_trades", "sell_trades"]
    showImageFeatures(imgs[40], titles, filename)
    ask = A._get_ask_price_features(A._mds[0].values)
    bid = A._get_bid_price_features(A._mds[0].values)
    print ask[0][:10]
    print bid[0][:10]


def _test4():
    feature_kargs = {
        "symbol": "TX_G0.TWF", 
        "colo": "twf",
        "path": "../data",
        "data_format": "h5",
        "tick_size": 1.,
        "n_levels": 5,
        "n_target_levels": 3,
        "look_back": 32,
        "window": 12,
        "cutoff": 500
    }
    A = GetFeatures(**feature_kargs)
    feature_df = A.get_features(n_threads=8)
    print "dataFrame shape: ", feature_df.shape
    print "column shapes:"
    for col in feature_df.columns:
        print col, ": ", np.array(feature_df[col].tolist()).shape
    
if __name__ == "__main__":
    ## _test1()
    ## _test2()
    ## _test3()
    _test4()