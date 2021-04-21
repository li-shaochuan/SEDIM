import os
import warnings
import tempfile

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

import tensorflow.keras as keras
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Dense, Dropout, Input,BatchNormalization
from keras.callbacks import EarlyStopping
import keras.losses
import keras
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_distance_matrix(raw, n_pred=None):
    VMR = raw.std() / raw.mean()
    VMR[np.isinf(VMR)] = 0

    if n_pred is None:
        potential_pred = raw.columns[VMR > 0]
    else:
        print("Using {} predictors".format(n_pred))
        potential_pred = VMR.sort_values(ascending=False).index[:n_pred]

    covariance_matrix = pd.DataFrame(np.abs(np.corrcoef(raw.T.loc[potential_pred])),
                                     index=potential_pred,
                                     columns=potential_pred).fillna(0)
    return covariance_matrix


def wMSE(y_true, y_pred, binary=False):
    if binary:
        weights = tf.cast(y_true > 0, tf.float32)
    else:
        weights = y_true
    return tf.reduce_mean(weights * tf.square(y_true - y_pred))


def inspect_data(data):
    # Check if there area any duplicated cell/gene labels

    if sum(data.index.duplicated()):
        print("ERROR: duplicated cell labels. Please provide unique cell labels.")
        exit(1)

    if sum(data.columns.duplicated()):
        print("ERROR: duplicated gene labels. Please provide unique gene labels.")
        exit(1)

    max_value = np.max(data.values)
    if max_value < 10:
        print("ERROR: max value = {}. Is your data log-transformed? Please provide raw counts"
              .format(max_value))
        exit(1)

    print("Input dataset is {} cells (rows) and {} genes (columns)"
          .format(*data.shape))
    print("First 3 rows and columns:")
    print(data.iloc[:3, :3])


class MultiNet:

    def __init__(self,
                 ncores=-1,
                 output_prefix=None,
                 sub_outputdim=512,
                 verbose=0,
                 seed=1234,
                 architecture=None
                 ):
        self.NN_parameters = architecture
        if self.NN_parameters.get('sub_outputdim') == None:
            self.sub_outputdim = sub_outputdim
        else:
            self.sub_outputdim = self.NN_parameters.get('sub_outputdim')
        self.outputdir = output_prefix
        self.verbose = verbose
        self.seed = seed
        self.setCores(ncores)

    def setCores(self, ncores):
        if ncores > 0:
            self.ncores = ncores
        else:
            self.ncores = os.cpu_count()
            print("Using all the cores ({})".format(self.ncores))


    def save(self, model):
        os.system("mkdir -p {}".format(self.outputdir))

        model_json = model.to_json()

        with open("{}/model.json".format(self.outputdir), "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        model.save_weights("{}/model.h5".format(self.outputdir))
        print("Saved model to disk in {}".format(self.outputdir))

    def save_to_path(self, path):
        os.system("mkdir -p {}".format(path))

        # model_json = self.model.to_json()
        #
        # with open("{}/model.json".format(path), "w") as json_file:
        #     json_file.write(model_json)

        # serialize weights to HDF5
        self.model.save("{}/model.h5".format(path))
        print("Saved model to disk in {}".format(path))

    def build(self, inputdims):
        print(self.NN_parameters)
        keras.backend.clear_session()
        inputs = [Input(shape=(inputdim,)) for inputdim in inputdims]
        outputs = inputs

        for i in range(self.NN_parameters.get('nb_layers')):
            outputs = [
                Dense(self.NN_parameters['nb_neurons'][i], activation=self.NN_parameters['activation'][i],

                      )(output)
                for output in outputs]
            if self.NN_parameters.get('dropout_rate')[i] != 0:
                outputs = [Dropout(self.NN_parameters['nb_neurons'][i], seed=self.seed)(output)
                           for output in outputs]
            if self.NN_parameters.get('BN')[i] != 0:
                outputs = [BatchNormalization()(output)
                           for output in outputs]
        outputs = [Dense(self.sub_outputdim, activation=self.NN_parameters['output_activation'])(output)
                   for output in outputs]

        model = Model(inputs=inputs, outputs=outputs)

        loss = self.NN_parameters['loss']
        print(loss)
        if loss in [k for k, v in globals().items() if callable(v)]:
            # if loss is a defined function
            loss = eval(self.NN_parameters['loss'])

        if not callable(loss):
            # it is defined in Keras
            if hasattr(keras.losses, loss):
                loss = getattr(keras.losses, loss)
            else:
                print('Unknown loss: {}. Aborting.'.format(loss))
                exit(1)
        optim = self.get_optimizer()
        model.compile(optimizer=optim,
                      loss=loss)
        return model

    def get_optimizer(self):
        if self.NN_parameters['optim'] == 'Adam':
            return keras.optimizers.Adam(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'SGD':
            return keras.optimizers.SGD(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'Adadelta':
            return keras.optimizers.Adadelta(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'Adagrad':
            return keras.optimizers.Adagrad(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'RMSprop':
            return keras.optimizers.RMSprop(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'NAdam':
            return keras.optimizers.Nadam(lr=self.NN_parameters['learning_rate'])
        elif self.NN_parameters['optim'] == 'Adamax':
            return keras.optimizers.Adamax(lr=self.NN_parameters['learning_rate'])

    def fit(self,
            raw,
            cell_subset=1,
            NN_lim=None,
            genes_to_impute=None,
            n_pred=None,
            ntop=5,
            minVMR=0.5,
            mode='random',
            ):

        inspect_data(raw)

        if self.seed is not None:
            np.random.seed(self.seed)

        if cell_subset != 1:
            if cell_subset < 1:
                raw = raw.sample(frac=cell_subset)
            else:
                raw = raw.sample(cell_subset)

        gene_metric = (raw.var() / (1 + raw.mean())).sort_values(ascending=False)
        gene_metric = gene_metric[gene_metric > 0]

        if genes_to_impute is None:
            genes_to_impute = self.filter_genes(gene_metric, minVMR, NN_lim=NN_lim)
        else:
            # Make the number of genes to impute a multiple of the network output dim
            n_genes = len(genes_to_impute)
            if n_genes % self.sub_outputdim != 0:
                print("The number of input genes is not a multiple of {}. Filling with other genes.".format(n_genes))
                fill_genes = gene_metric.index[:self.sub_outputdim - n_genes]

                if len(fill_genes) < self.sub_outputdim - n_genes:
                    # Not enough genes in gene_metric. Sample with replacement
                    rest = self.sub_outputdim - n_genes - len(fill_genes)
                    fill_genes = np.concatenate([fill_genes,
                                                 np.random.choice(gene_metric.index, rest, replace=True)])

                genes_to_impute = np.concatenate([genes_to_impute, fill_genes])

        covariance_matrix = get_distance_matrix(raw, n_pred=n_pred)

        self.setTargets(raw.reindex(columns=genes_to_impute), mode=mode)
        self.setPredictors(covariance_matrix, ntop=ntop)

        print("Normalization")
        norm_data = np.log1p(raw).astype(np.float32)  # normalizer.transform(raw)

        np.random.seed(self.seed)
        tf.random.set_random_seed(self.seed)

        tf.config.threading.set_inter_op_parallelism_threads(self.ncores)
        tf.config.threading.set_intra_op_parallelism_threads(self.ncores)

        print("Building network")
        model = self.build([len(genes) for genes in self.predictors])

        test_cells = np.random.choice(norm_data.index, int(0.05 * norm_data.shape[0]), replace=False)
        train_cells = np.setdiff1d(norm_data.index, test_cells)

        X_train = [norm_data.loc[train_cells, inputgenes].values for inputgenes in self.predictors]
        Y_train = [norm_data.loc[train_cells, targetgenes].values for targetgenes in self.targets]

        X_test = [norm_data.loc[test_cells, inputgenes].values for inputgenes in self.predictors]
        Y_test = [norm_data.loc[test_cells, targetgenes].values for targetgenes in self.targets]

        print("Fitting with {} cells".format(norm_data.shape[0]))
        result = model.fit(X_train, Y_train,
                           validation_data=(X_test, Y_test),
                           epochs=self.NN_parameters["epoch"],
                           batch_size=self.NN_parameters["batch_size"],
                           callbacks=[EarlyStopping(monitor='val_loss',
                                                    patience=5)],
                           verbose=self.verbose)

        self.trained_epochs = len(result.history['loss'])
        print("Stopped fitting after {} epochs".format(self.trained_epochs))

        self.model=model

        # Save some metrics on test data
        # Y_test_raw = np.hstack(Y_test).flatten()
        # Y_test_imputed = np.hstack(model.predict(X_test)).flatten()
        #
        # # Keep only positive values (since negative values could be dropouts)
        # Y_test_imputed = Y_test_imputed[Y_test_raw > 0]
        # Y_test_raw = Y_test_raw[Y_test_raw > 0]
        #
        # self.test_metrics = {
        #     'correlation': pearsonr(Y_test_raw, Y_test_imputed)[0],
        #     'MSE': np.sum((Y_test_raw - Y_test_imputed) ** 2) / len(Y_test_raw)
        # }

        return self

    def predict(self,
                raw,
                imputed_only=False,
                policy="restore"):

        norm_raw = np.log1p(raw)

        inputs = [norm_raw.loc[:, predictors].values.astype(np.float32)
                  for predictors in self.predictors]

        model = self.model

        predicted = model.predict(inputs)
        if len(inputs) > 1:
            predicted = np.hstack(predicted)

        predicted = pd.DataFrame(predicted, index=raw.index, columns=self.targets.flatten())

        predicted = predicted.groupby(by=predicted.columns, axis=1).mean()
        not_predicted = norm_raw.drop(self.targets.flatten(), axis=1)

        imputed = (pd.concat([predicted, not_predicted], axis=1)
                   .loc[raw.index, raw.columns]
                   .values)

        # To prevent overflow
        imputed[(imputed > 2 * norm_raw.values.max()) | (np.isnan(imputed))] = 0
        # Convert back to counts
        imputed = np.expm1(imputed)

        if policy == "restore":
            print("Filling zeros")
            mask = (raw.values > 0)
            imputed[mask] = raw.values[mask]
        elif policy == "max":
            print("Imputing data with 'max' policy")
            mask = (raw.values > imputed)
            imputed[mask] = raw.values[mask]

        imputed = pd.DataFrame(imputed, index=raw.index, columns=raw.columns)

        if imputed_only:
            return imputed.loc[:, predicted.columns]
        else:
            return imputed

    def filter_genes(self,
                     gene_metric,  # assumes gene_metric is sorted
                     threshold,
                     NN_lim=None
                     ):
        if not str(NN_lim).isdigit():
            NN_lim = (gene_metric > threshold).sum()

        n_subsets = int(np.ceil(NN_lim / self.sub_outputdim))
        genes_to_impute = gene_metric.index[:n_subsets * self.sub_outputdim]

        rest = self.sub_outputdim - (len(genes_to_impute) % self.sub_outputdim)

        if rest > 0:
            fill_genes = np.random.choice(gene_metric.index, rest)
            genes_to_impute = np.concatenate([genes_to_impute, fill_genes])

        print("{} genes selected for imputation".format(len(genes_to_impute)))

        return genes_to_impute

    def setTargets(self, data, mode='random'):

        n_subsets = int(data.shape[1] / self.sub_outputdim)

        if mode == 'progressive':
            self.targets = data.columns.values.reshape([n_subsets, self.sub_outputdim])
        else:
            self.targets = np.random.choice(data.columns,
                                            [n_subsets, self.sub_outputdim],
                                            replace=False)

    def setPredictors(self, covariance_matrix, ntop=5):
        self.predictors = []

        for i, targets in enumerate(self.targets):

            genes_not_in_target = np.setdiff1d(targets, covariance_matrix.columns)

            if genes_not_in_target.size == 0:
                warnings.warn(
                    'Warning: number of target genes lower than output dim. Consider lowering down the sub_outputdim parameter',
                    UserWarning)
                genes_not_in_target = covariance_matrix.columns

            subMatrix = (covariance_matrix
                .loc[targets, genes_not_in_target]
                )
            sorted_idx = np.argsort(-subMatrix.values, axis=1)
            predictors = subMatrix.columns[sorted_idx[:, :ntop].flatten()]

            self.predictors.append(predictors.unique())

            print("Net {}: {} predictors, {} targets"
                  .format(i, len(np.unique(predictors)), len(targets)))

    def score(self, data, policy=None):
        warnings.warn(
            "This method is deprecated. Please use model.test_metrics to measure model accuracy instead",
            DeprecationWarning)
        Y_hat = self.predict(data, policy=policy)
        Y = data.loc[Y_hat.index, Y_hat.columns]

        return pearsonr(Y_hat.values.reshape(-1), Y.values.reshape(-1))





