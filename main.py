
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras import Model
from test import *
from keras.layers import Input, Dense,Dropout
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import regularizers
from keras.utils import plot_model
import numpy as np
from keras.models import load_model
gpu_id = '1,3'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction =1
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))
from util import *
from model import classfier
import tensorflow as tf

def train(nn_param,x_train, x_test, y_train, y_test):
    tf.reset_default_graph()
    multi = classfier(architecture=nn_param)
    acc=multi.fit(x_train, y_train,x_test,  y_test)
    return acc,multi
def cross_valadition(x,y,nn_param):
        print("cross valadition --------------------")
        print(nn_param)
        k_size = 10
        k_fold = StratifiedKFold(k_size, True, random_state=len(y))
        index = k_fold.split(X=x, y=np.argmax(y,axis=1))
        acc_all=0
        for train_index, test_index in index:
            x_train = (x[train_index])
            x_test = (x[test_index])
            y_train = y[train_index]
            y_test = (y[test_index])

            acc,model=train(nn_param,x_train, x_test, y_train, y_test)
            acc_all+=acc
        print("acc:{}".format(acc_all/k_size))

        return acc_all / k_size
def Ant(x,y,n_iterations,pop_size,n_best):
    best_model=None
    best_fitness=0
    from ACP import AntColony
    from ArchParameter import Generator
    pop=Generator().create_Random_netpop(pop_size)
    fitness=[]

    for i in range (pop_size):

        fit,model=cross_valadition(x,y,pop[i])
        fitness.append(fit)
        if fit>best_fitness:
            best_fitness=fit
            best_model=pop[i]
    ev_optim=AntColony(pop_size,n_best,nn_keys=pop[0].keys())
    for i in range(1,n_iterations+1):
        print('At {}-th iteration the best fitness is :{}'.format(i-1,best_fitness))
        ev_optim.spread_pheronome(pop,fitness)
        pop=ev_optim.gen_path()
        for ind in range (pop_size):
            fit,model = cross_valadition(x, y, pop[ind])
            fitness[ind]=fit
            ev_optim.local_updating_rule(pop[ind],fit)
            if fit > best_fitness:
                best_fitness = fit
                best_model =pop[ind]
    return best_fitness,best_model

import pandas as pd
def main():
    n_iterations = 5
    pop_size = 20
    n_best = 2
    path='datapath'
    h5 = pd.HDFStore(r'{}/x.h5'.format(path)
                     , mode='r')
    x = h5.get('x')
    h5.close()
    h5 = pd.HDFStore(r'{}/y.h5'.format(path)
                     , mode='r')
    y = h5.get('y')
    h5.close()
    X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.1,random_state=0)
    index=gene_selection(X_train,y_train)
    X_train=X_train.iloc[:,index]
    X_test=X_test.iloc[:,index]
    one_hot = np.eye(4).tolist()
    temp_y = list(np.array(y_train).squeeze())
    y_one_hot_train = []
    for i in range(len(temp_y)):
        j = temp_y[i]
        y_one_hot_train.append(one_hot[j])
    temp_y = list(np.array(y_test).squeeze())
    y_one_hot_test = []
    for i in range(len(temp_y)):
        j = temp_y[i]
        y_one_hot_train.append(y_one_hot_test[j])

    fit, model= Ant(np.array(X_train),np.array(y_one_hot_train), n_iterations, pop_size, n_best)
    print('final fitness:{}'.format(fit))
    print('best model:{}'.format(model))

    acc,model=train(model, X_train, X_test, y_train, y_test)
    print('test_acc:{}'.format(acc))
if __name__ == "__main__":
    main()









