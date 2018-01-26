## Necessary 
from gc_sampler import GetFeatures
import numpy as np
import pandas as pd

# Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
plt.switch_backend('agg')

## Shuffling
from sklearn.cross_validation import train_test_split

## Keras
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten
from keras.layers import Flatten, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping#, TensorBoard
from keras import backend as K
from keras import optimizers
from keras.optimizers import SGD
from keras import initializers
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

## Initializing mpl
mpl.rc('font', family = 'serif', size = 17)
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.minor.size'] = 2

# Function to compute class weights
def comp_cls_wts(y, pwr = 0.5):
    '''
    Used to compute class weights
    '''
    dic = {}
    for x in set(y):
        dic[x] = len(y)**pwr/list(y).count(x)**pwr
    return dic

# Convert dataFrame into Keras data format
def get_inout_data(dataset):
    X = {
        'ask': np.array(dataset.askp.tolist()).astype(np.int32),
        'bid': np.array(dataset.bidp.tolist()).astype(np.int32),
        'img': np.array(dataset.img.tolist()).astype(np.float32),
        'y': np_utils.to_categorical(dataset.behav.values, num_classes=9)
    }
    return X
    
if __name__ == '__main__':
    symbol ='TX_G0.TWF'
    colo = 'twf'
    dates = []
    tick_size = 1.
    gf = GetFeatures(symbol=symbol, colo=colo, dates=dates, tick_size=tick_size)
    gf.readRawData(path = '../data')
    df = gf.getTrainDataSet()
    print df.shape
    
    dtrain, dvalid = train_test_split(df, random_state=17, train_size=0.7)
    cls_wts = comp_cls_wts(dtrain.behav.values, pwr=0.5)
    print cls_wts
    
    tr = get_inout_data(dtrain)
    cv = get_inout_data(dvalid)
    MAX_PRICE = np.max([np.max(vec) for vec in tr['ask']] + [np.max(vec) for vec in cv['ask']]) + 5
    
    ## Input layers
    ask = Input(shape=[tr['ask'][0].shape[0]], name = 'ask')
    bid = Input(shape=[tr['bid'][0].shape[0]], name = 'bid')
    img = Input(shape=(6, 10, 4), name = 'img')
    
    ## RNN part
    emb_ask = Embedding(MAX_PRICE, 64)(ask)
    emb_bid = Embedding(MAX_PRICE, 64)(bid)
    rnn_ask = GRU(32)(emb_ask)
    rnn_bid = GRU(32)(emb_bid)
    fc_ask = Dropout(0.1) (Dense(128) (rnn_ask))
    fc_bid = Dropout(0.1) (Dense(128) (rnn_bid))
    
    ## CNN part
    cnn_img = Conv2D(32, kernel_size = (3, 3), padding = 'same', activation='relu') (img)
    cnn_img = Dropout(0.1)(cnn_img)
    cnn_img = Conv2D(64, kernel_size = (3, 3), padding = 'same', activation='relu') (cnn_img)
    cnn_img = Dropout(0.1)(cnn_img)
    fc_img = Flatten()(cnn_img)
    fc_img = Dropout(0.1) (Dense(128) (fc_img))
    
    ## Main layer
    main = concatenate([
        fc_ask,
        fc_bid,
        fc_img
    ])
    main = Dropout(0.25) (Dense(256) (main))
    main = Dropout(0.5) (Dense(48) (main))
    output = Dense(9, activation="softmax") (main)
    
    model = Model([ask, bid, img], output)
    model.summary()
    
    N_epoch = 60
    learning_rate = 0.03
    decay_rate = 1./1.20
    optimizer = SGD(learning_rate)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    ### Train the model
    print("TRAINING BEGINS!")
    
    ## Using adaptive decaying learning rate
    def scheduler(epoch):
        global learning_rate
        global decay_rate
        if epoch%10 == 0:
            learning_rate *= decay_rate
            print("CURRENT LEARNING RATE = ", learning_rate)
        return learning_rate
    change_lr = LearningRateScheduler(scheduler)
    
    ## Train the model
    res = model.fit([tr['ask'], tr['bid'], tr['img']], tr['y'], batch_size = 128, epochs = N_epoch, 
                    verbose = 1, validation_data = ([cv['ask'], cv['bid'], cv['img']], cv['y']), 
                    class_weight = cls_wts, callbacks = [change_lr])
    
    ## Plot results
    steps = [i for i in range(N_epoch)]
    train_accu = res.history['acc']
    train_loss = res.history['loss']
    test_accu = res.history['val_acc']
    test_loss = res.history['val_loss']
    
    print("VISUALIZATION:")
        ## Plotting the results
    fig, axes = plt.subplots(2,2, figsize = (12, 12))
    fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

    axes[0][0].set_title('Loss')
    axes[0][0].plot(steps, train_loss, label = 'train loss')
    axes[0][0].plot(steps, test_loss, label = 'test loss')
    axes[0][0].set_xlabel('# of steps')
    axes[0][0].legend()

    axes[0][1].set_title('Accuracy')
    axes[0][1].plot(steps, train_accu, label = 'train accuracy')
    axes[0][1].plot(steps, test_accu, label = 'test accuracy')
    axes[0][1].set_xlabel('# of steps')
    axes[0][1].legend()
    
    plt.savefig('convrg_rst.png')
    
                    
    