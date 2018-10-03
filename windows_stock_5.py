import pandas as pd
import numpy as np 
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Flatten,Dense

#Encoding the integer sequences into a binary matrix


def vectorize_sequences(sequences,dimension=28):
	results=np.zeros((len(sequences),dimension))
	for i,sequence in enumerate(sequences):
		results[i,sequence]=1.
	return results


def data_preparation(dataframe):
	dff = dataframe[0:999]

	return dff

#Load and Preparation of truck data



df_skoupa=pd.read_csv('window_skoupa.csv')

dff_skoupa = data_preparation(df_skoupa)

skoupa_dataframe=dff_skoupa

labels_skoupa=np.ones((998,),dtype=int)


#Load and Preparation of ix data


df_ix=pd.read_csv('window_ix.csv')

dff_ix = data_preparation(df_ix)

ix_dataframe=dff_ix

labels_ix=np.zeros((998,),dtype=int)



#Tables merge


frames = [ix_dataframe,skoupa_dataframe]
merge = pd.concat(frames)
merge_labels = np.concatenate((labels_ix,labels_skoupa),axis=0)


#Metatropi Dataframe se Array


merge = merge.values


#Model Definition


model=models.Sequential()


model.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(125,)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))


#Compiling the model and configuring optimizer,loss and metrics


model.compile(optimizer=optimizers.Nadam(),loss='binary_crossentropy',metrics=[metrics.binary_accuracy])


#Training the model


history=model.fit(merge,merge_labels,epochs=100,batch_size=64)


