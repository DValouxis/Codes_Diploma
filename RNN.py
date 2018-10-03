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
from keras.preprocessing import sequence
from keras.layers import Embedding,SimpleRNN,LSTM
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

skoupa_dataframe=dff_skoupa

labels_skoupa=np.ones((5800,),dtype=int)


#Load and Preparation of ix data


df_ix=pd.read_csv('ix_vid-137.csv')

dff_ix = data_preparation(df_ix)

ix_dataframe=dff_ix

labels_ix=np.zeros((5800,),dtype=int)



#Tables merge


frames = [ix_dataframe,skoupa_dataframe]
merge = pd.concat(frames)
merge_labels = np.concatenate((labels_ix,labels_skoupa),axis=0)


#Metatropi Dataframe se Array


merge = merge.values


#Model Definition


merge=sequence.pad_sequences(merge,maxlen=27)
model=models.Sequential()

model.add(Embedding(27,64))
#model.add(layers.Dropout(0.2))
model.add(SimpleRNN(64))
#model.add(layers.Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))


#Compiling the model and configuring optimizer,loss and metrics


model.compile(optimizer=optimizers.Nadam(),loss='binary_crossentropy',metrics=[metrics.binary_accuracy])


#Training the model


history=model.fit(merge,merge_labels,epochs=30,batch_size=512)


