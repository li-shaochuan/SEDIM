
import scipy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import csv
from BBO import BBO
import pandas as pd
import scipy.io
import matplotlib as mpl
import numpy as np
import keras
mpl.use('Agg')
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
mpl.use('Agg')
import tensorflow as tf
from generateArtich import *

gpu_id = '2,0,1,3'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction =1
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))


from multinet_bbo import MultiNet
import os

def load_GSE102827(PATH):
    if os.path.exists(PATH+'/GSE102827_MATRIX.h5'):
        h5 = pd.HDFStore(PATH+'/GSE102827_MATRIX.h5', mode='r')
        data = h5.get('truth')
        h5.close()
        h5 = pd.HDFStore(PATH + '/GSE102827_label.h5', mode='r')
        label = h5.get('label')
        h5.close()
    else:
        data = pd.read_csv(
            PATH+'/GSE102827_MATRIX.csv',
            index_col=0, header=0
        )
        print(data)
        label = pd.read_csv(
            PATH+'/GSE102827_cell_type_assignments.csv',
            index_col=0, header=0
        )
        print(data)
        print(label)
        label=label.loc[:,'celltype']
        label=label.dropna(axis=0, how='any')
        print(label)
        data=data.T.loc[label.index,:]
        print(data)
        data.to_hdf(path_or_buf=PATH+'/GSE102827_MATRIX.h5',key='truth')
        label.to_hdf(path_or_buf=PATH+'/GSE102827_label.h5',key='label')
    return data,label
def SEDIM(**kwargs):
    pop_size = 10
    generation_size = 10
    pop = list()
    missing_rate=0.1
    Elite_size=2
    pmutate=0.2
    threshold_fitness=0.5
    print('Load DATA')

    path = [r"/home/data" 
                  ]
    dataset_name=['GSE102827']
    all_data_true = dict()
    for i in range(len(path)):
        all_data_true[dataset_name[i]],label=load_GSE102827(path[i])
        
            

    all_data_masked = dict()


    from mask import dataMask

    for i in range(len(dataset_name)):
        if os.path.exists(
                r'/home/SEDIM/scRNAmasked_data/{}.h5'.format(dataset_name[i]+ '_N_MASKED_PER_CELL10' )):

            print('reading {}'.format(dataset_name[i] + '_N_MASKED_PER_CELL10'))

            h5=pd.HDFStore(r'/home/SEDIM/scRNAmasked_data/{}.h5'.format(dataset_name[i]
                                                                                   +'_N_MASKED_PER_CELL10')
                           , mode='r')
            all_data_masked[dataset_name[i]]=h5.get('raw')
            h5.close()
            # all_data_masked[dataset_name[i]] = pd.read_csv(
            #     r'/home/SEDIM/scRNAmasked_data/{}.csv'.format(dataset_name[i] + '_N_MASKED_PER_CELL10'),
            #     index_col=0, header=0
            # )
            print(all_data_masked[dataset_name[i]])
        else:
            print('masking:{}'.format(dataset_name[i]))
            raw_ = dataMask(np.array(all_data_true[dataset_name[i]]),
                            all_data_true[dataset_name[i]].index, all_data_true[dataset_name[i]].columns)

            raw_.to_hdf(path_or_buf=r'/home/SEDIM/scRNAmasked_data/{}.h5'.format(dataset_name[i] +'_N_MASKED_PER_CELL10'),
                        key='raw')

            # raw_.to_csv(
            #     r'/home/SEDIM/scRNAmasked_data/{}.csv'.format(dataset_name[i] +'_N_MASKED_PER_CELL10'))
            print(raw_)
            all_data_masked[dataset_name[i]] = raw_
    print('read Down')
    use_defatlutNet=False
    from generateArtich import Generator
    net_dict_Generator=Generator()

    best_fitness_for_dataset=dict()
    import sys
    for key in all_data_true.keys():
            truth_ = all_data_true[key].copy()

            raw_ = all_data_masked[key].copy()
            best_fitness_for_dataset[key]=sys.maxsize
            #创建种群
            print('parameters setting')
            if use_defatlutNet:
                pop=net_dict_Generator.create_defalutNet_Pop(pop_size)
            else:
                pop = net_dict_Generator.create_Random_netpop(pop_size)

            fitness = list()
            # set the initial fitness
            for i in range(pop_size):
                keras.backend.clear_session()
                truth = truth_
                raw = raw_.copy()
                NN_params = pop[i]
                print(NN_params)
                output_dir = '/home/SEDIM/model/{}'.format(key + str(i))
                multi = MultiNet(output_prefix=output_dir, architecture=NN_params)
                multi.fit(raw, NN_lim="auto", cell_subset=1, minVMR=0.5, n_pred=None)
                # create metrics
                mask = (raw != truth)
                df_SEDIM = multi.predict(raw, imputed_only=True)
                #print(df_SEDIM)
                gene_subset = df_SEDIM.columns
                truth = truth.reindex(columns=gene_subset)
                mask = mask.reindex(columns=gene_subset)
                df_SEDIM = df_SEDIM.values[mask.values]
                truth = truth.values[mask.values]

                mseScore = mean_squared_error(np.log1p(truth),
                                              np.log1p(df_SEDIM))
                pearsonScore = pearsonr(np.log1p(truth),
                                        np.log1p(df_SEDIM))[0]
                print('mse:{},pearson{}'.format(mseScore, pearsonScore))
                fitness.append(mseScore)
                #store the best net
                if mseScore<best_fitness_for_dataset[key]:
                    best_fitness_for_dataset[key]=mseScore
                    df_SEDIM = multi.predict(raw, imputed_only=True)
                    df_SEDIM.to_hdf(path_or_buf=r'/home/SEDIM/impute/{}.h5'.format(
                        key + '_N_MASKED_PER_CELL10_impute_only_V2'),
                        key='impute')
                    df_SEDIM_all = multi.predict(raw, imputed_only=False)
                    df_SEDIM_all.to_hdf(path_or_buf=r'/home/SEDIM/impute/{}.h5'.format(
                        key + '_N_MASKED_PER_CELL10_impute_all_V2'),
                        key='impute')
                    #multi.save_to_path(best_net_storge_path)
            fitness_storage=list(fitness)
            ind_storage=list(pop)
            BBO_optim = BBO(Elite_size, pop_size, pop[0].keys(), pmutate)
            from fitnessPredict import fitnessPredict
            fitnessPredictor=fitnessPredict()
            #进化
            print('performing the BBO process')
            for iter in range(generation_size):

                #种群更新
                print('before the process of BBO:{}'.format(fitness))
                pop_BBO,EliteSolution,EliteCost=BBO_optim.evolue(list(pop),list(fitness))
                #pop reseted, reset the fitness
                #caculate the fitness of pop after the BBO
                fitness_predict_by_ml=fitnessPredictor.predict(list(pop_BBO),list(ind_storage),list(fitness_storage))

                #greedy_strategy
                for i in range(len(fitness_predict_by_ml)):
                    if fitness_predict_by_ml[i]<fitness[i]:
                        print('Calculate objective function for reseted individual under greedy strategy')
                        keras.backend.clear_session()
                        truth = truth_
                        raw = raw_.copy()
                        NN_params = pop_BBO[i]
                        print(NN_params)
                        output_dir = '/home/SEDIM/modelgithub/BBO_V2{}'.format(key + str(i))
                        multi = MultiNet(output_prefix=output_dir, architecture=NN_params)
                        multi.fit(raw, NN_lim="auto", cell_subset=1, minVMR=0.5, n_pred=None)
                        # create metrics
                        mask = (raw != truth)
                        df_SEDIM = multi.predict(raw, imputed_only=True)
                        #print(df_SEDIM)
                        gene_subset = df_SEDIM.columns
                        truth = truth.reindex(columns=gene_subset)
                        mask = mask.reindex(columns=gene_subset)
                        df_SEDIM = df_SEDIM.values[mask.values]
                        truth = truth.values[mask.values]

                        mseScore = mean_squared_error(np.log1p(truth),
                                                      np.log1p(df_SEDIM))
                        pearsonScore = pearsonr(np.log1p(truth),
                                                np.log1p(df_SEDIM))[0]
                        print('mse:{},pearson{}'.format(mseScore, pearsonScore))
                        fitness_local = mseScore
                        if fitness_local<fitness[i]:
                            fitness[i]=fitness_local
                            pop[i]= pop_BBO[i]
                        ind_storage.append(pop_BBO[i])
                        fitness_storage.append(fitness_local)
                        # store the best net
                        if mseScore < best_fitness_for_dataset[key]:
                            best_fitness_for_dataset[key] = mseScore
                            df_SEDIM = multi.predict(raw, imputed_only=True)
                            df_SEDIM.to_hdf(path_or_buf=r'/home/SEDIM/impute/{}.h5'.format(
                                key + '_N_MASKED_PER_CELL10_impute_only__V2'),
                                key='impute')
                            df_SEDIM_all = multi.predict(raw, imputed_only=False)
                            df_SEDIM_all.to_hdf(path_or_buf=r'/home/SEDIM/impute/{}.h5'.format(
                                key + '_N_MASKED_PER_CELL10_impute_all_V2'),
                                key='impute')
                            #multi.save_to_path(best_net_storge_path)
                print('after the process of BBO:{}'.format(fitness))

                pop,fitness=BBO_optim.sort(list(pop),list(fitness))
                # Replacing the individual of population with EliteSolution
                for k in range(Elite_size):
                    pop[(pop_size - 1) - k] = EliteSolution[k]
                    fitness[(pop_size - 1) - k] = EliteCost[k]

                # Removing the duplicate individuals
                pop,reset_ind_index=BBO_optim.remove_Dup(pop)
                #After the remove_DUP process, the pop has changed.
                #Calculate objective function for each individual
                for i in range(len(reset_ind_index)):
                        print('Calculate objective function for reseted individual')
                        keras.backend.clear_session()
                        truth = truth_
                        raw = raw_.copy()
                        NN_params = pop[reset_ind_index[i]]
                        print(NN_params)
                        output_dir = '/home/SEDIM/modelgithub/BBO_V2{}'.format(key + str(i))
                        multi = MultiNet(output_prefix=output_dir, architecture=NN_params)
                        multi.fit(raw, NN_lim="auto", cell_subset=1, minVMR=0.5, n_pred=None)
                        #create metrics
                        mask = (raw != truth)
                        df_SEDIM = multi.predict(raw, imputed_only=True)
                        #print(df_SEDIM)
                        gene_subset = df_SEDIM.columns
                        truth = truth.reindex(columns=gene_subset)
                        mask = mask.reindex(columns=gene_subset)
                        df_SEDIM=df_SEDIM.values[mask.values]
                        truth=truth.values[mask.values]

                        mseScore = mean_squared_error(np.log1p(truth),
                                                      np.log1p(df_SEDIM))
                        pearsonScore = pearsonr(np.log1p(truth),
                                                np.log1p(df_SEDIM))[0]
                        print('mse:{},pearson{}'.format(mseScore,pearsonScore))
                        fitness[reset_ind_index[i]]=mseScore
                        ind_storage.append(pop[reset_ind_index[i]])
                        fitness_storage.append(mseScore)
                        # store the best net
                        if mseScore < best_fitness_for_dataset[key]:
                            best_fitness_for_dataset[key] = mseScore
                            df_SEDIM = multi.predict(raw, imputed_only=True)
                            df_SEDIM.to_hdf(path_or_buf=r'/home/SEDIM/impute/{}.h5'.format(
                                key + '_N_MASKED_PER_CELL10_impute_only__V2'),
                                key='impute')
                            df_SEDIM_all = multi.predict(raw, imputed_only=False)
                            df_SEDIM_all.to_hdf(path_or_buf=r'/home/SEDIM/impute/{}.h5'.format(
                                key + '_N_MASKED_PER_CELL10_impute_all_V2'),
                                key='impute')
                            #multi.save_to_path(best_net_storge_path)
                #Sort the population on fitness
                pop, fitness = BBO_optim.sort(pop, fitness)

                print(['At iteration ' + str(iter) + ' the best fitness is ' + str(fitness[0])]);
                print('The best network parameters:{}'.format(pop[0]))
                print('dataset_name:{}'.format(key))


if __name__ == "__main__":
    SEDIM()