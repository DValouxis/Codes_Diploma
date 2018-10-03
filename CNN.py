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
	df = dataframe[3:5803]
	driving_qua = df['driving_qualities'].str.split(",",expand=True).add_prefix('driving_')
	driving_qua=driving_qua.astype('int')
	dff=pd.concat([df[['wln_accel_max','wln_brk_max','wln_crn_max']],driving_qua],1)
	return dff


#Load and Preparation of truck data


df_skoupa=pd.read_csv('skoupa_vid-27.csv')

dff_skoupa = data_preparation(df_skoupa)

#Train_skoupa

train_data_skoupa = dff_skoupa[:4603]
#print train_data_skoupa
train_labels_skoupa=np.ones((4603),dtype=int)

#Validate_skoupa

val_data_skoupa = dff_skoupa[4603:5803]
val_labels_skoupa=np.ones((1197),dtype=int)

#Load and Preparation of ix data


df_ix=pd.read_csv('ix_vid-137.csv')

dff_ix = data_preparation(df_ix)

#Train_ix

train_data_ix = dff_ix[:4603]
train_labels_ix=np.zeros((4603),dtype=int)

#Validate_ix

val_data_ix = dff_ix[4603:5803]
val_labels_ix=np.zeros((1197),dtype=int)

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


#Merge One Hot

merge_train_onehot=to_categorical(merge_train_data)
print merge_train_onehot.shape
merge_train_onehot=merge_train_onehot.reshape((9206,24,25,1))

merge_val_onehot=to_categorical(merge_val_data)
print merge_val_onehot.shape
merge_val_onehot=merge_val_onehot.reshape((2394,24,18,1))



merge_tlabels_onehot = np.asarray(merge_train_labels).astype('float32')
merge_vlabels_onehot = np.asarray(merge_val_labels).astype('float32')


#Model Definition


model=models.Sequential()


model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(24,25,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))




#Compiling the model and configuring optimizer,loss and metrics


model.compile(optimizer=optimizers.Nadam(),loss='binary_crossentropy',metrics=[metrics.binary_accuracy])


#Training the model


history=model.fit(merge_train_onehot,merge_tlabels_onehot,epochs=30,batch_size=512,validation_data=(merge_val_onehot,merge_vlabels_onehot))



