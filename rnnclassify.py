import re
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import load_files


# Functions
#############
def doc2wv(instr):
    '''
    Converts sentences to word vectors
    (first reduces all to lower case)
    '''
    res=[]
    for word in (tmp.group().strip() for tmp in wordreg.finditer(instr.lower())):
        try:
            res.append(worddict[word])
        except KeyError:
            pass
    return np.array(res)


# Params
##########
trainpath='../Naili_Data/traindata/'
wordvecpath='/data/glove/glove.6B.50d.txt'
wordreg=re.compile("\S+")
nUnits=128
embedDim=50

# Loading in data
###################
dat=load_files(trainpath)
df=pd.read_csv(wordvecpath,sep="\s",skiprows=[8],names=['word']+['d'+str(tmp) for tmp in range(50)])
worddict=dict(zip(df.word,df.iloc[:,1:].values))


# Creating the network
########################
x=tf.placeholder(tf.float32,(None,None,embedDim))
cell=tf.contrib.rnn.BasicLSTMCell(nUnits)
rnnLayer=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)


# Run it!
sess=tf.Session()
