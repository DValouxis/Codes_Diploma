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


def data_preparation(dataframe):
	df = dataframe[0:10000]
	driving_qua = df['driving_qualities'].str.split(",",expand=True).add_prefix('driving_')
	driving_qua=driving_qua.astype('int')
	dff=pd.concat([df[['speed','wln_accel_max','wln_brk_max','wln_crn_max']],driving_qua],1)
	return dff


def acceleration_summary(dff_skoupa):
	sum=np.zeros(10000).astype(int)
	k=0
	for i in range(0,10000):
		sum[i]=0             
		k=0 
		for j in range(14,25):
			#print i,j
			sum[i] = sum[i] + dff_skoupa[i][j] * pow(10,k)
			#print dff_skoupa[i][j],sum,pow(10,k)
			k=k+1

	#print sum 
	return sum

def braking_summary(dff_skoupa):
	sum = np.zeros(10000).astype(int)
	
	k=0
	for i in range(0,10000):
		sum[i]=0
		k=0 
		for j in range(4,14):
			#print i,j
			sum[i] = sum[i] + dff_skoupa[i][j] * pow(10,10-k)
			#print dff_skoupa[i][j],sum,pow(10,k)
			k=k+1

	#print sum 
	return sum


#def rpm(dff_skoupa):
	#return rrpm

def speed(dff_skoupa):
	rspeed = dff_skoupa[:,0]
	return rspeed

def row_mean(dff_skoupa):
	mean=np.zeros(10000)
	for i in range(0,10000):
		mean[i] = np.mean(dff_skoupa[i,:])
	return mean

def max_acc(dff_skoupa):
	max_ac = dff_skoupa[:,1]
	return max_ac

def max_bra(dff_skoupa):
	max_br = dff_skoupa[:,2]
	return max_br


def row_var(dff_skoupa):
	variance=np.zeros(10000)
	for i in range(0,10000):
		variance[i] = np.var(dff_skoupa[i,:])
	return variance

#Load and Preparation of truck data


df_skoupa=pd.read_csv('ix_vid-137.csv')

dff_skoupa = data_preparation(df_skoupa)

#print dff_skoupa,dff_skoupa.shape
dff_skoupa = dff_skoupa.values
print dff_skoupa,dff_skoupa.shape

#Acceleration_summary

acceleration_s = acceleration_summary(dff_skoupa).astype(float)
acceleration_s = acceleration_s.reshape(10000,1)
print acceleration_s

#Braking_summary

braking_s = braking_summary(dff_skoupa).astype(float)
braking_s = braking_s.reshape(10000,1)
print braking_s

#Speed
 
rspeed = speed(dff_skoupa).astype(float)
rspeed = rspeed.reshape(10000,1)
print rspeed

#Mean

mean = row_mean(dff_skoupa).astype(float)
mean = mean.reshape(10000,1)
print mean

#Max_acceleration

max_acce = max_acc(dff_skoupa).astype(float)
max_acce = max_acce.reshape(10000,1)
print max_acce

#Max_braking

max_br = max_bra(dff_skoupa).astype(float)
max_br = max_br.reshape(10000,1)
print max_br

#Variance

variance = row_var(dff_skoupa).astype(float)
variance = variance.reshape(10000,1)
print variance




final = np.concatenate((acceleration_s,braking_s),axis=1)
final = np.concatenate((final,rspeed),axis = 1)
final = np.concatenate((final,mean),axis = 1)
final = np.concatenate((final,max_acce),axis = 1)
final = np.concatenate((final,max_br),axis = 1)
final = np.concatenate((final,variance),axis = 1)
print final

np.savetxt("feautures_val_ix.csv", final, delimiter=",")







