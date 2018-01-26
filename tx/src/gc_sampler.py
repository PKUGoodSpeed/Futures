'''
Convert Raw market data into trainable features
The raw market data is directly sampled from gsampler
Generating a huge data frame including features
So far, the train target is other participants' behavior
Other people's behavior has the following classes:
(Only considerting the operations on the top levels)
    Send an aggressive buy order (0)
    Send an aggressive sell order (1)
    Send Market making buy order (2)
    Send Market making sell order (3)
    Send a passive buy order (4)
    Send a passive sell order (5)
    Cancel a passive buy order (6)
    Cancel a passive sell order (7)
    Do nothing (8)
'''
## Math and data handler
import numpy as np
import pandas as pd

## display dataFrame
from IPython.display import display

## Recording execution time
import time
import os

## Using multi threading 
import multiprocessing
import multiprocessing.pool

NUM_THREADS = 8
class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
    
START_TIME = "09:00:00"
END_TIME = "13:30:00"
LOOK_BACK = 6
NUM_LEVELS = 10

''' ============= Extracting other people's behavior ============='''
def _getLabels(prev, post):
    if prev[9] != 0:
        return None
    if post[9] == 1:
        return [0]
    if post[9] == -1:
        return [1]
    target =[]
    if prev[0] != post[0] or prev[4] != post[4]:
        if post[0] > prev[0]:
            target.append(2)
        elif post[0] < prev[0]:
            target.append(3)
        if post[4] > prev[4]:
            target.append(2)
        elif post[4] < prev[4]:
            target.append(3)
        return list(set(target))
    elif post[1] != prev[1] or post[5] != prev[5]:
        if post[1] > prev[1]:
            target.append(5)
        elif post[1] < prev[1]:
            target.append(7)
        if post[5] > prev[5]:
            target.append(4)
        elif post[5] < prev[5]:
            target.append(6)
        return list(set(target))
    else:
        return [8]
def _getLabelsWrapper(args):
    return _getLabels(*args)
    
''' ============= Extracting our own features ============='''
'''
1. Imgs feature with 4 channels:
    - ask book sizes
    - bid book sizes
    - aggressive buy trade
    - aggressive sell trade
2. Ask price level for RNN
3. Bid price level for RNN
'''
def _getImage(vecs):
    assert len(vecs) == LOOK_BACK
    img = np.zeros((LOOK_BACK, NUM_LEVELS, 4))
    ask_base = int(vecs[5][0] - int(NUM_LEVELS/2))
    bid_base = int(vecs[5][4] - int(NUM_LEVELS/2) + 1)
    for i in range(LOOK_BACK):
        l = vecs[i][0] - ask_base
        if l>=0 and l<NUM_LEVELS:
            img[i][l][0] = vecs[i][1]
        l = vecs[i][2] - ask_base
        if l>=0 and l<NUM_LEVELS:
            img[i][l][0] = vecs[i][3]
        l = vecs[i][4] - bid_base
        if l>=0 and l<NUM_LEVELS:
            img[i][l][1] = vecs[i][5]
        l = vecs[i][6] - bid_base
        if l>=0 and l<NUM_LEVELS:
            img[i][l][1] = vecs[i][7]
        if vecs[i][9] > 0.5:
            l = vecs[i][8] - ask_base
            if l>=0 and l<NUM_LEVELS:
                img[i][l][2] = vecs[i][10]
        if vecs[i][9] < -0.5:
            l = vecs[i][8] - bid_base
            if l>=0 and l<NUM_LEVELS:
                img[i][l][3] = vecs[i][10]
    return img
    
def _getRnnAskPrice(vecs):
    assert len(vecs) == LOOK_BACK
    return vecs[:, 0]

def _getRnnBidPrice(vecs):
    assert len(vecs) == LOOK_BACK
    return vecs[:, 4]

    
class GetFeatures:
    '''
    Convert the raw market data into trainable features
    '''
    _dates = None
    _symbol = None
    _colo = None
    _tick_size = None
    _md_dfs = None     ## A list of dataFrame
    _cutoff = 20
    
    def __init__(self, symbol, colo, dates, tick_size):
        ''' No need explain '''
        self._symbol = symbol
        self._colo = colo
        self._dates = dates
        self._tick_size = tick_size
    
    def readRawData(self, path = '../data'):
        print("Start reading market data...")
        start_time = time.time()
        self._md_dfs = []
        self._dates = []
        for f in os.listdir(path):
            filename = path + "/" + f
            df = pd.read_csv(filename)
            self._md_dfs.append(df)
            self._dates.append(f.split('.')[1])
        print("Reading data finished!")
        print("Reading data time usage: " + str(time.time() - start_time) + " s")
        
    
    def _getBehaviors(self, mat):
        ''' Extracting other people's behavior '''
        print("Extracting other people's behaviors...")
        args = [(mat[i+6], mat[i+7]) for i in range(len(mat) - self._cutoff)]
        start_time = time.time()
        pool = MyPool(NUM_THREADS)
        targets = pool.map(_getLabelsWrapper, args)
        pool.close()
        pool.join()
        print("Extracting finished!")
        print("Extracting time: " + str(time.time() - start_time) + " s")
        return targets
    
    def _getImageFeature(self, mat):
        ''' Extracting image features '''
        print("Extracting image features...")
        args = [np.array(mat[i:i+6]) for i in range(len(mat) - self._cutoff)]
        for arg in args:
            assert arg.shape == (6, 11)
        start_time = time.time()
        pool = MyPool(NUM_THREADS)
        imgs = np.array(pool.map(_getImage, args))
        pool.close()
        pool.join()
        print("Extracting finished!")
        print("Extracting time: " + str(time.time() - start_time) + " s")
        print(imgs.shape)
        return imgs
        
    def _getRnnFeature(self, mat):
        ''' Extracting Rnn features '''
        print("Extracting Rnn features...")
        args = [np.array(mat[i:i+6]) for i in range(len(mat) - self._cutoff)]
        start_time = time.time()
        pool = MyPool(NUM_THREADS)
        asks = np.array(pool.map(_getRnnAskPrice, args))
        pool.close()
        pool.join()
        pool = MyPool(NUM_THREADS)
        bids = np.array(pool.map(_getRnnBidPrice, args))
        pool.close()
        pool.join()
        print("Extracting finished!")
        print("Extracting time: " + str(time.time() - start_time) + " s")
        print(asks.shape)
        print(bids.shape)
        return asks, bids
        
    def getTrainDataSet(self):
        ''' Combing all features '''
        print("Generating training sets...")
        start_time = time.time()
        f_asks = []
        f_bids = []
        f_imgs = []
        f_target = []
        for date, df in zip(self._dates, self._md_dfs):
            print("\n***** Processing data for {0} *****".format(date))
            mat = (df.values*(1.00001)/self._tick_size).astype(np.int32)
            target = self._getBehaviors(mat)
            imgs = self._getImageFeature(mat)*self._tick_size
            asks, bids = self._getRnnFeature(mat)
            
            for i in range(len(target)):
                if target[i] is None:
                    continue
                for x in target[i]:
                    f_asks.append(np.array(asks[i]))
                    f_bids.append(np.array(bids[i]))
                    f_imgs.append(np.array(imgs[i]))
                    f_target.append(int(x))
        print(np.array(f_asks).shape)
        print(np.array(f_bids).shape)
        print(np.array(f_imgs).shape)
        print(np.array(f_target).shape)
        print("Job finished!")
        print("Time usage: " + str(time.time() - start_time) + " s")
        return pd.DataFrame({
            'img': f_imgs,
            'askp': f_asks,
            'bidp': f_bids,
            'behav': f_target
        })
        
            
            
    
    def test(self):
        df = pd.DataFrame(self._md_dfs[0])
        mat = (df.values*(1.00001)/self._tick_size).astype(np.int32)
        asks, bids = self._getRnnFeature(mat)
        print mat[:10]
        print '\n'
        print asks[:3]
        print bids[:3]
        
    

if __name__ == '__main__':
    symbol ='TX_G0.TWF'
    colo = 'twf'
    dates = []
    tick_size = 1.
    gf = GetFeatures(symbol=symbol, colo=colo, 
                    dates=dates, tick_size=tick_size)
    ## gf.sampleRawData()
    gf.readRawData(path = '../data')
    ## gf.test()
    df = gf.getTrainDataSet()
    print df.shape
    display(df[:6])
        
