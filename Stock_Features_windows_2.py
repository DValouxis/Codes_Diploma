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
	dff = dataframe[0:2499]

	return dff





#Load and Preparation of skoupa 

#Train_data

df_skoupa=pd.read_csv('window_skoupa.csv')

dff_skoupa = data_preparation(df_skoupa)

train_data_skoupa = dff_skoupa[:2000]

train_labels_skoupa=np.ones((2000),dtype=int)

#Validate_skoupa

val_data_skoupa = dff_skoupa[2000:2499]

val_labels_skoupa=np.ones((497),dtype=int)




#Load and Preparation of ix data


df_ix=pd.read_csv('window_ix.csv')

dff_ix = data_preparation(df_ix)

#Train_ix

train_data_ix = dff_ix[:2000]

train_labels_ix=np.zeros((2000),dtype=int)


#Validate_ix

val_data_ix = dff_ix[2000:2499]
val_labels_ix=np.zeros((497),dtype=int)



#Tables merge train

frames_train = [train_data_ix,train_data_skoupa]
merge_train_data = pd.concat(frames_train)
merge_train_labels = np.concatenate((train_labels_ix,train_labels_skoupa),axis=0)



#Table merge validate

frames_val = [val_data_ix,val_data_skoupa]
merge_val_data = pd.concat(frames_val)
merge_val_labels = np.concatenate((val_labels_ix,val_labels_skoupa),axis=0)

#Metatropi Dataframe se Array


merge_train_data = merge_train_data.values
merge_val_data = merge_val_data.values



#Model Definition


model=models.Sequential()


model.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(14,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))


#Compiling the model and configuring optimizer,loss and metrics


model.compile(optimizer=optimizers.Nadam(),loss='binary_crossentropy',metrics=[metrics.binary_accuracy])


#Training the model


history=model.fit(merge_train_data,merge_train_labels,epochs=30,batch_size=512,validation_data=(merge_val_data,merge_val_labels))


