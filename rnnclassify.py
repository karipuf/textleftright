import tensorflow as tf
from sklearn.datasets import load_files


# Params
trainpath='../Naili_Data/traindata/'
wordvecpath='/data/glove/glove.6B.50d.txt'

# Loading in data
dat=load_files(trainpath)
df=pd.read_csv(wordvecpath,sep="\s",skiprows=[8],names=['word']+['d'+str(tmp) for tmp in range(50)])
worddict=dict(zip(df.word,df.iloc[:,1:].values))

# Converting to word vectors
def doc2wv(instr):
    return -1
