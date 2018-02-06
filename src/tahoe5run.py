## basic
import numpy as np
import pandas as pd
import json
## sampler
from sampler import sample_market_data
## features
from features import GetFeatures
## model
from models import KerasModel
## option parser
import optsparser
## Spliting train and valid
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    instr_cfg_file, model_cfg_file = optsparser.getopts()
    instr_cfg = json.load(open(instr_cfg_file))
    model_cfg = json.load(open(model_cfg_file))
    print("Instrument options: ")
    print(instr_cfg)
    print("Model options: ")
    print(model_cfg)
    
    ## Sampling data
    if instr_cfg["resample"]:
        kargs = dict({
            "symbol": instr_cfg["symbol"],
            "colo": instr_cfg["colo"],
            "n_levels": instr_cfg["n_levels"],
            "data_path": instr_cfg["data_path"],
            "n_threads": instr_cfg["n_threads"],
            "save": instr_cfg["data_format"]
        }, **instr_cfg["sampler"])
        sample_market_data(**kargs)
    
    ## Geting Features
    kargs = dict({
        "symbol": instr_cfg["symbol"],
        "colo": instr_cfg["colo"],
        "n_levels": instr_cfg["n_levels"],
        "path": instr_cfg["data_path"],
        "data_format": instr_cfg["data_format"]
    }, **instr_cfg["feature"])
    feature_obj = GetFeatures(**kargs)
    feature_df = feature_obj.get_features(n_threads=instr_cfg["n_threads"])
    print "dataFrame shape: ", feature_df.shape
    print "column shapes:"
    for col in feature_df.columns:
        print col, ": ", np.array(feature_df[col].tolist()).shape
        
    ## Trainning with particular model
    train, valid = train_test_split(feature_df, train_size=model_cfg["split_ratio"])
    train_x = np.array(train.images.tolist())
    train_y = np.array(train.target.tolist())
    valid_x = np.array(valid.images.tolist())
    valid_y = np.array(valid.target.tolist())
    
    print train_x.shape
    print train_y.shape
    print valid_x.shape
    print valid_y.shape
    
    
    keras_model = KerasModel(input_shape=train_x[0].shape, output_dim=len(train_y[0]))
    model_kargs = model_cfg["model"]
    model = keras_model.getModel(model_kargs["model_type"], **model_kargs["kargs"])
    model.summary()
    history = keras_model.train(train_x, train_y, valid_x, valid_y, **model_cfg["train"])
    
    output_file="{0}_convergence.png".format(model_cfg['model_name'])
    keras_model.plot(model_cfg['output_dir'] + '/' + output_file)