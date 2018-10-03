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
	dff = dataframe[0:1000]

	return dff





#Load and Preparation of truck data


df_skoupa=pd.read_csv('window_skoupa.csv')

dff_skoupa = data_preparation(df_skoupa)

#Train_skoupa
train_size = 0.20*len(dff_skoupa)
train_size = int(train_size)

train_data_skoupa = dff_skoupa[:train_size]
train_labels_skoupa = np.ones((train_size),dtype=int)

#Validate_skoupa
val_size = 0.40*len(dff_skoupa)
val_size = int(val_size)

val_data_skoupa = dff_skoupa[train_size:train_size+val_size]
val_labels_skoupa = np.ones((val_size),dtype=int)

#Test_skoupa
test_size = 0.40*len(dff_skoupa)
test_size = int(test_size)

test_data_skoupa = dff_skoupa[train_size+val_size:len(dff_skoupa)]

test_labels_skoupa = np.ones((test_size+1),dtype=int)


#Load and Preparation of ix data


df_ix=pd.read_csv('window_ix.csv')

dff_ix = data_preparation(df_ix)

#Train_ix

train_data_ix = dff_ix[:train_size]
train_labels_ix=np.zeros((train_size),dtype=int)

#Validate_ix

val_data_ix = dff_ix[train_size:train_size+val_size]
val_labels_ix = np.zeros((val_size),dtype=int)

#Test_ix

test_data_ix = dff_ix[train_size+val_size:len(dff_skoupa)]
test_labels_ix = np.zeros((test_size+1),dtype=int)

#Tables merge train

frames_train = [train_data_ix,train_data_skoupa]
merge_train_data = pd.concat(frames_train)
merge_train_labels = np.concatenate((train_labels_ix,train_labels_skoupa),axis=0)

#Table merge validate

frames_val = [val_data_ix,val_data_skoupa]
merge_val_data = pd.concat(frames_val)
merge_val_labels = np.concatenate((val_labels_ix,val_labels_skoupa),axis=0)

#Table merge test

frames_test = [test_data_ix,test_data_skoupa]
merge_test_data = pd.concat(frames_test)
merge_test_labels = np.concatenate((test_labels_ix,test_labels_skoupa),axis=0)

#Metatropi Dataframe se Array


merge_train_data = merge_train_data.values
merge_val_data = merge_val_data.values
merge_test_data = merge_test_data.values


#Model Definition


model=models.Sequential()


model.add(layers.Dense(64,activation='relu',input_shape=(50,)))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))


#Compiling the model and configuring optimizer,loss and metrics


model.compile(optimizer=optimizers.Nadam(),loss='binary_crossentropy',metrics=[metrics.binary_accuracy])


#Training the model


history=model.fit(merge_train_data,merge_train_labels,epochs=30,batch_size=512,validation_data=(merge_val_data,merge_val_labels))

test_loss,test_accu = model.evaluate(merge_test_data,merge_test_labels)
print test_accu

