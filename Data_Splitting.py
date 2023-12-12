# Options for splitting
# Functions for train/test, train/test/val/etc
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import numpy as np




class split_data:
    def __init__(self, train_x, train_y, test_x=None, test_y=None,val_x=None,val_y=None,query_x=None,query_y=None):
        self.trainx = train_x
        self.trainy = train_y
        self.testx = test_x
        self.testy = test_y
        self.valx = val_x
        self.valy = val_y
        self.queryx = query_x
        self.queryy = query_y
        if(train_x is not None and train_y is not None and test_x is not None and test_y is not None and val_x is not None and val_y is not None and query_x is not None and query_y is not None):
            self.t = "train_test_val_query"
        elif(train_x is not None and train_y is not None and test_x is not None and test_y is not None and val_x is not None and val_y is not None):
            self.t = "train_test_val"
        elif(train_x is not None and train_y is not None and test_x is not None and test_y is not None):
            self.t = "train_test"
        elif(train_x is not None and train_y is not None):
            self.t = "train"
        else:
            self.t = "improper format"
            
    def resample(self):
        # oversample connected neuron pairs
        ros = RandomOverSampler(random_state=0)
        # Oversample but with all features
        self.trainx, self.trainy = ros.fit_resample(self.trainx,self.trainy)
    
def train(data):
    train_x = data.loc[:, data.columns != "connected"]
    train_y = data["connected"]
    return (split_data(train_x,train_y))

def train_test(data, test_ratio=0.2):
    train_data, test_data = train_test_split(data, test_size=test_ratio, random_state=14)
    train_x = train_data.loc[:, train_data.columns != "connected"]
    train_y = train_data["connected"]
    test_x = test_data.loc[:, test_data.columns != "connected"]
    test_y = test_data["connected"]
    
    return (split_data(train_x,train_y,test_x,test_y))

def train_test_val(data, test_ratio=0.2, val_ratio=0.2):

    train_ratio = 1-test_ratio-val_ratio

    train_data, test_data = train_test_split(data, test_size=1 - train_ratio)
    val_data, test_data = train_test_split(test_data, test_size=test_ratio/(test_ratio + val_ratio)) 
    
    train_x = train_data.loc[:, train_data.columns != "connected"]
    train_y = train_data["connected"]
    test_x = test_data.loc[:, test_data.columns != "connected"]
    test_y = test_data["connected"]
    val_x = val_data.loc[:, test_data.columns != "connected"]
    val_y = val_data["connected"]
    
    return (split_data(train_x,train_y,test_x,test_y,val_x,val_y))