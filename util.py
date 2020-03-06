# Modules
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import array as ar
import os
from operator import itemgetter
import h5py

# Class files
from LSTM import * 
from LSTMbatch import *

def fit_H_to_LSTM(data_split, path):

    targetfolder=listdir('measurements/'+path)
    datapaths={}

    batchStack = {}
    namemat=[]
    filelist=[]

    ### skapar alla paths
    for x in range(0,len(targetfolder)):

        datapaths[targetfolder[x]+'path']='measurements/'+path+targetfolder[x]
        # dict with all paths to the folders

        globals()[targetfolder[x]]=os.listdir(datapaths[targetfolder[x]+'path'])
        # list of all files in targeted folder

        sorted(globals()[targetfolder[x]])                  # sort the files
        filelist.append(globals()[targetfolder[x]])
        namemat.append(targetfolder[x]+'mat')                # målet är [halfmat, ...

    n_files = len(globals()[targetfolder[0]])           # number of files in the folder

    matris=[]


    for i in range(0,n_files):
        matris2=[]
        for x in range(0,len(targetfolder)):


            temp1=h5py.File(itemgetter(targetfolder[x]+'path')(datapaths)+'/'+filelist[x][i],'r')
            temp2=temp1.get('acc')
            matris.append(temp2[1,:])

            if i/n_files <= data_split[0]/100:
                    batchStack.update({
                        'batch'+str(i) : LongShortTermMemoryBatch(matris,
                                         i,     
                                         category = 'train')
                         })

            elif i/n_files > data_split[0]/100 and i/n_files <= (data_split[0]+data_split[1])/100:
                    batchStack.update({
                        'batch'+str(i) : LongShortTermMemoryBatch(matris,
                                        i,
                                        category = 'validation')
                        })

            else:
                batchStack.update({
                        'batch'+str(i) : LongShortTermMemoryBatch(matris,
                                        i,
                                        category = 'test')
                        })
            return batchStack    
        matris=[]
                                                    
def save_model(model,name):
    '''
    https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    '''
    model_json = model.to_json()
    with open('models/'+name+'.json', 'w') as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights('models/'+name+'.h5')
    print('Saved model:', name)

7
