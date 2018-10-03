from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import pandas as pd
import numpy as np 
import tensorflow as tf


def train_input_fn(features, labels,batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    #print dataset	
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(20).repeat().batch(batch_size)
    #print dataset
    return dataset

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset



def main(self):

    #Load prep truck

    df_truck=pd.read_csv('skoupa_vid-27.csv')
    #print df_truck
    dff_truck=df_truck[3:5003]
    #print(len(dff_truck))
    dff_truck, label_truck = dff_truck, dff_truck.pop('Labels')
    #print labels_kappa,type(labels_kappa)


    driving_qua_truck=dff_truck['driving_qualities'].str.split(",",expand=True).add_prefix('driving_')
    driving_qua_truck=driving_qua_truck.astype('float64')


    dff_truck=pd.concat([dff_truck[['wln_accel_max','wln_brk_max','wln_crn_max']],driving_qua_truck],1)
    dff_truck=dff_truck.astype('float64')
    #print dff_truck,type(dff_truck)
    #print labels_truck

    #Load and Preparation of ix data


    df_ix=pd.read_csv('ix_vid-137.csv')

    dff_ix=df_ix[3:5003]

    dff_ix, label_ix= dff_ix ,dff_ix.pop('Labels')
    #print labels_ix,dff_ix

    driving_qua_ix=dff_ix['driving_qualities'].str.split(",",expand=True).add_prefix('driving_')
    driving_qua_ix=driving_qua_ix.astype('int')


    dff_ix=pd.concat([dff_ix[['wln_accel_max','wln_brk_max','wln_crn_max']],driving_qua_ix],1)
    dff_ix=dff_ix.astype('float64')
    #print dff_ix


    #Preparation of validation Data and Training Data

    val_truck = dff_truck[4003:5003]
    
    train_truck = dff_truck[3:4002]
    
    labels_truck=label_truck[3:4002]

    #print(len(labels_truck))
    #print(len(labels_ix))
    val_ix = dff_ix[4003:5003]
    
    train_ix = dff_ix[3:4002]
    #train_labels_ix = np.zeros((3,),dtype=int)
    labels_ix=label_ix[3:4002]
    #print(len(labels_ix))
    labels_ix_val=label_ix[4003:5003]
    #print(len(labels_ix_val))
    labels_truck_val=label_truck[4003:5003]

    #Tables merge for Training Data
    #print train_ix

    frames_train = [train_ix,train_truck]
    #print frames_train
    merge_train = pd.concat(frames_train)
    merge_train_final = merge_train

    #Tables merge for Validation Data

    frames_val = [val_ix,val_truck]
    merge_val = pd.concat(frames_val)
    merge_val_final = merge_val

    #Normalization

    merge_train_final = merge_train_final.astype(float)

    mean =merge_train_final.mean(axis=0) 
    merge_train_final -= mean
    std= merge_train_final.std(axis=0) 
    for i in range(len(std)):
	    if(std[i] == 0):
		    std[i]=1
    merge_train_final /= std

    merge_val_final = merge_val_final.astype(float)

    mean =merge_val_final.mean(axis=0)
    merge_val_final -= mean
    std= merge_val_final.std(axis=0)
    for i in range(len(std)):
	    if(std[i] == 0):
		    std[i]=1
    merge_val_final /= std

    #Labels merge for Training Data

    frame_labels=[labels_ix,labels_truck]
    merge_train_labels = pd.concat(frame_labels)

    frame_labels_val=[labels_ix_val,labels_truck_val]
    merge_val_labels=pd.concat(frame_labels_val)

    my_feature_columns = []
    for key in merge_train_final.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))


    
    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,hidden_units = [128,128,128,128],n_classes=2)

    classifier.train(input_fn=lambda:train_input_fn(merge_train_final,merge_train_labels,256),steps=1000)


    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda:eval_input_fn(merge_val_final,merge_val_labels, 256))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)













