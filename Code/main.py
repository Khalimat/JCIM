#  Imports
from __future__ import print_function

#  Importing general-purpose modules
from pathlib import Path
import pandas as pd
import numpy as np

import argparse
import time

#  Importing modules from RDKit and DeepChem
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import deepchem as dc

#  Importing modules from abc package
from abc import ABC, abstractmethod

#  Importing modules from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

#  Importing modules from scipy
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import gmean

#  Importing the modules from imbalanced learning
from imblearn.over_sampling import ADASYN, SMOTE  # up-sampling
from imblearn.under_sampling import CondensedNearestNeighbour, InstanceHardnessThreshold  # down-sampling

#  Importing the modules for active learning
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, entropy_sampling

#  Importing modules from TensorFlow and Keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

#  Save model 
import h5py
import io
import copy

#  Importing modules to filter out warnings
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")


#  Defining variables to automatically run the pipeline

#  Up- and down-sampling
SAMPLING = {'N': False,
            'SMOTE': SMOTE(),
            'ADASYN': ADASYN(),
            'CNN': CondensedNearestNeighbour(),
            'IHT': InstanceHardnessThreshold()}

def sampl(X_train, Y_train, sampling):
    """Computes midranks.
    Args:
       X_train
          np.array, training features
       Y_train
          np.array, responses
       sampling
          trained sampling approach

    Returns:
       X_train
          transformed np.array with training features
       Y_train
          transformed np.array with responses
    """
    sl = sampling
    X_train, Y_train = sl.fit_resample(X_train, Y_train)
    return X_train, Y_train

#  Dataset
DATASETS = {'SF': 'SCAMS_filtered.csv',
            'SP1': 'SCAMS_balanced_with_positive.csv',
            'SP2': 'SCAMS_added_positives_653_1043.csv'}
#  AL query mode
SELECTION_MODE = {'uncertainty_sampling': uncertainty_sampling,
                  'classifier_entropy': entropy_sampling}
#  Metrics to calculate
METRICS = ['AUC_LB_test', 'AUC_test', 'AUC_UB_test', 'Accuracy_test',
           'F1_test', 'MCC_test', 'AUC_LB_validation', 'AUC_validation',
           'AUC_UB_validation', 'Accuracy_validation', 'F1_validation',
           'MCC_validation']

#  A list of columns for tables with run statistics
col_statis = ['Iteration', 'AUC_LB_test', 'AUC_test',
              'AUC_UB_test', 'Accuracy_test', 'F1_test', 'MCC_test',
              'AUC_LB_validation', 'AUC_validation',
              'AUC_UB_validation', 'Accuracy_validation',
              'F1_validation', 'MCC_validation']


class KerasClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    """

    def __getstate__(self):
        state = self.__dict__
        if "model" in state:
            model = state["model"]
            model_hdf5_bio = io.BytesIO()
            with h5py.File(model_hdf5_bio, mode="w") as file:
                model.save(file)
            state["model"] = model_hdf5_bio
            state_copy = copy.deepcopy(state)
            state["model"] = model
            return state_copy
        else:
            return state

    def __setstate__(self, state):
        if "model" in state:
            model_hdf5_bio = state["model"]
            with h5py.File(model_hdf5_bio, mode="r") as file:
                state["model"] = tf.keras.models.load_model(file)
        self.__dict__ = state


def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2

def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight=None):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    sample_weight = None
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)

def calc_auc_ci(y_true, y_pred, alpha=0.95):
    auc, auc_cov = delong_roc_variance(y_true, y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1
    return auc, ci

#  Code to run butina classtering
def butina_cluster(mol_list, cutoff=0.35):
    """
    Reference
    __________
    @article{butina1999unsupervised,
      title={Unsupervised data base clustering based on daylight's fingerprint and Tanimoto similarity:
      A fast and automated way to cluster small and large data sets},
      author={Butina, Darko},
      journal={Journal of Chemical Information and Computer Sciences},
      volume={39},
      number={4},
      pages={747--750},
      year={1999},
      publisher={ACS Publications}
    }
    Make a clusters based Butina clustering algorithm using MorganFingerprints similarity

    Function adopted from:
    https://github.com/PatWalters/Learning_Cheminformatics/blob/master/clustering.ipynb

    Parameters
    __________
    :param mol_list: list of rdkit.Chem.rdchem.Mol object to cluster
    :param cutoff: Tanimoto similarity cutoff

    Return
    _______
    :return: list with cluster id for every molecule from mol_list

    """
    fp_list = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mol_list]
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1 - x for x in sims])
    mol_clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_id_list = [0] * nfps
    for idx, cluster in enumerate(mol_clusters, 1):
        for member in cluster:
            cluster_id_list[member] = idx
    return cluster_id_list

def _generate_scaffold(smiles, include_chirality=False):
    """
    I copied the function from deepchem.
    Compute the Bemis-Murcko scaffold for a SMILES string.
    Bemis-Murcko scaffolds are described in DOI: 10.1021/jm9602928.
    They are essentially that part of the molecule consisting of
    rings and the linker atoms between them.

    Paramters
    ---------
    smiles: str
    SMILES
    include_chirality: bool, default False
    Whether to include chirality in scaffolds or not.
    Returns
    -------
    str
    The MurckScaffold SMILES from the original SMILES
    References
    ----------
    .. [1] Bemis, Guy W., and Mark A. Murcko. "The properties of known drugs.
     1. Molecular frameworks." Journal of medicinal chemistry 39.15 (1996): 2887-2893.
    Note
    ----
    This function requires RDKit to be installed.: bool
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    except ModuleNotFoundError:
        raise ImportError("This function requires RDKit to be installed.")

    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

#  Code to generate scaffolds for scaffolds-based split function
def generate_scaffolds(dataset):
    """Returns all scaffolds from the dataset.
    Adapted from deepchem
    Parameters
    ----------
    dataset: Dataset
      Dataset to be split.
    log_every_n: int, optional (default 1000)
      Controls the logger by dictating how often logger outputs
      will be produced.
    Returns
    -------
    scaffold_sets: List[List[int]]
      List of indices of each scaffold in the dataset.
    """
    scaffolds = {}
    data_len = dataset.shape[0]

    for ind, smiles in enumerate(dataset['Smiles String']):
        scaffold = _generate_scaffold(smiles)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets

#  Utility functions to run the pipeline
def str2bool(v):
    """
    Transfer string to bool
    Parameter
    ----------
    v: str
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def rm_tree(pth):
    """
    Recursively remove files from the directory

    Parameter
    ----------
    pth: Pathlib path

    """
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def file_doesnot_exist(path, file_name):
    """
    Returns full path if file exist, else raises error

    Parameters
    ----------
    file_name: str, file name
    path: str, directory

    Returns
    -------
    full_file_path: Path object

    """

    full_file_path = Path(path) / file_name
    if not full_file_path.is_file():
        raise FileNotFoundError(
            'File {} not found in location {}. Please, inter valid path and file name'.format(file_name, path))
    else:
        return full_file_path

def bool_to_strint(b_v):
    if b_v:
        return 'True'
    else:
        return 'False'

def describe(mols):
    """
    Function from DeepSCAMs approach

    Parameters
    ----------
    mols: list of mol-files

    Returns
    -------
    descrs: list with descriptor
            space for molecules
    """
    featurizer1 = dc.feat.CircularFingerprint()
    X1 = featurizer1([m for m in mols])
    featurizer2 = dc.feat.RDKitDescriptors()
    X2 = featurizer2([m for m in mols])
    X = np.hstack((X1, X2))
    return X

def f_one_mcc_score(model, X_test, Y_test):
    """
    Calculate F1-score and Matthews correlation coefficient (MCC)

    :param model:trained model
    :param X_test: test samples to predict labels
    :param Y_test: true labels
    :return: F1-score, MCC
    """
    y_pred = model.predict(X_test)
    f_one = f1_score(Y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
    return f_one, (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5


class SCAMsPipeline:
    def __init__(self, study_name, datasets_path,
                 ID_name='ID', iterations=10,
                 validation_name='test_DLS.csv',
                 X_column_name='Smiles String',
                 Y_column_name='agg?'):
        self.iterations = iterations
        self.study_name = study_name
        self.sampling = SAMPLING[self.study_name.split('_')[0]]
        self.run_sampling = [False if not self.sampling else True][0]
        dataset_name = DATASETS[self.study_name.split('_')[1]]
        self.splitter = SPLIT[self.study_name.split('_')[2]]

        self.results_dir = Path.cwd().parents[0] / 'Results' / self.study_name

        train_test_dataset = pd.read_csv(file_doesnot_exist(datasets_path, dataset_name))
        train_validation_dataset = pd.read_csv(file_doesnot_exist(datasets_path, validation_name))
        if Path.is_dir(self.results_dir):
            out_srt = input('The path exists. Do you want to delete the existing path and create the new? Please, enter yes or no: ')
            delete_existing_path = str2bool(out_srt)
            if delete_existing_path:
                rm_tree(self.results_dir)
            else:
                new_study = input('Please, enter new name: ')
                self.results_dir = Path.cwd() / 'Results' / new_study
        Path.mkdir(self.results_dir)

        self.train_test_dataset = Dataset(train_test_dataset,
                                          ID_name, X_column_name, Y_column_name)
        self.validation_dataset = Dataset(train_validation_dataset,
                                          ID_name, X_column_name, Y_column_name)
        self.DeepSCAMs_non_AL_res = None
        self.TF_non_AL_res = None
        self.TF_AL_res = None
        self.run_pipeline()

    def run_pipeline(self):
        """
        Run the pipeline
        """
        for i in range(self.iterations):
            fold = Run(i, self.validation_dataset.X, self.validation_dataset.Y,
                       self.train_test_dataset, self.splitter,
                       self.sampling, self.run_sampling, self.results_dir)
            self.DeepSCAMs_non_AL_res = self.make_df_with_stats(fold.perf_stats_deepscams_non_AL, self.DeepSCAMs_non_AL_res)
            self.TF_non_AL_res = self.make_df_with_stats(fold.perf_stats_tensorflow_non_AL, self.TF_non_AL_res)
            self.TF_AL_res = self.make_df_with_stats(fold.perf_stats_tensorflow_AL, self.TF_AL_res)
        self.DeepSCAMs_non_AL_res.to_csv(self.results_dir / 'DeepSCAMs.csv')
        self.TF_non_AL_res.to_csv(self.results_dir / 'TF_ML_non_AL.csv')
        self.TF_AL_res.to_csv(self.results_dir / 'TF_ML_AL.csv')


    @staticmethod
    def make_df_with_stats(df_with_stats, performance_stats):
        """
        Make dataframe with performance stats

        Parameters
        ----------
        df_with_stats: A dataframe or None object with performance stats over passed iterations
        performance_stats: iteration stats


        """
        if df_with_stats is None:
            df_with_stats = pd.DataFrame(performance_stats,
                                         columns=col_statis)
        else:
            df_with_stats = pd.concat([df_with_stats,
                                       pd.DataFrame(performance_stats,
                                                    columns=col_statis
                                                    )])
        return df_with_stats



def dataset_to_splitter(splitter, dataset, type):
    """
    Feed the dataset to a chosen splitter
    """

    splitted = splitter(dataset.X, dataset.Y,
                        dataset.SMILES,
                        dataset.dataset.mols,
                        dataset.ID_name, type, 0.3)
    return splitted

class Run:
    def __init__(self, iteration, X_validation,
                 Y_validation, test_train_to_split,
                 splitter, sampling, run_sampling,
                 results_dir):
        self.iteration = iteration
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.test_train_to_split = test_train_to_split
        self.splitter = splitter
        self.run_sampling = run_sampling
        self.results_dir = results_dir
        self.sampling = sampling
        self.perf_stats_deepscams_non_AL = None
        self.perf_stats_tensorflow_non_AL = None
        self.perf_stats_tensorflow_AL = None
        self.run()

    def run(self):
        start = time.time()
        self.run_split()
        self.run_non_al()
        self.run_al()
        end = time.time()
        print('The run {} took {} minutes'.format(self.iteration, int((end - start)/60)))

    def run_split(self):
        train_test = dataset_to_splitter(self.splitter,
                                         self.test_train_to_split,
                                         '1')
        self.X_train = train_test.X_train
        self.Y_train = train_test.Y_train
        self.X_test = train_test.X_test
        self.Y_test = train_test.Y_test

    def run_non_al(self):
        if self.run_sampling:
            X_train, Y_train = sampl(self.X_train, self.Y_train, self.sampling)
        else:
            X_train, Y_train = self.X_train, self.Y_train

        DeepSCAMs = DeepSCAMsModel(X_train, Y_train, self.X_test,
                                   self.Y_test, self.X_validation, self.Y_validation)

        TensorFlow = TensorFlowModel(X_train, Y_train, self.X_test,
                                     self.Y_test, self.X_validation, self.Y_validation)

        DeepSCAMs_MLP_non_AL = [self.iteration] + DeepSCAMs.validation_performance.iloc[0].tolist() + DeepSCAMs.test_performance.iloc[0].tolist()
        TF_MLP_non_AL = [self.iteration] + TensorFlow.validation_performance.iloc[0].tolist() + TensorFlow.test_performance.iloc[0].tolist()
        self.perf_stats_deepscams_non_AL = pd.DataFrame([DeepSCAMs_MLP_non_AL], columns=col_statis)

        self.perf_stats_tensorflow_non_AL = pd.DataFrame([TF_MLP_non_AL],
                                                         columns=col_statis)


    def run_al(self):
        n_queries = int(0.88 * self.X_train.shape[0]) - 11
        TensorFlowAL = ALModel(self.X_train, self.Y_train,
                          self.X_test, self.Y_test,
                          self.X_validation, self.Y_validation,
                          n_queries, q_strategy=entropy_sampling,
                          iteration=self.iteration)
        TensorFlowAL.final_model.model.save(self.results_dir / 'TF_MLP_AL_{}.h5'.format(self.iteration))
        self.perf_stats_tensorflow_AL = pd.DataFrame([[self.iteration] + TensorFlowAL.validation_performance.iloc[0].tolist() +
                                                     TensorFlowAL.test_performance.iloc[0].tolist()],
                                                     columns=col_statis)


class Dataset:
    def __init__(self, dataset,
                 ID_name, X_column_name,
                 Y_column_name):
        self.dataset = dataset
        self.dataset = self.dataset.set_index([pd.Index(self.dataset[ID_name])])
        self.ID_name = self.dataset[ID_name]
        self.SMILES = self.dataset[X_column_name]
        self.Y_column_name = Y_column_name
        self.X = None
        self.Y = None
        self.run()

    def run(self):
        self.calculate_descriptors()

    def calculate_descriptors(self):
        mol_obj = [Chem.MolFromSmiles(s) for s in self.SMILES]
        self.dataset['mols'] = self.SMILES.apply(lambda x: Chem.MolFromSmiles(x))
        descriptors = describe(mol_obj)
        self.X = descriptors
        self.Y = np.array(self.dataset[self.Y_column_name])


class BaseSplitter:
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        self.X = X
        self.Y = Y
        self.SMILES = SMILES
        self.mols = mols
        self.ID = ID
        self.type = type
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.test_split_r = test_split_r
        self.split()

    @abstractmethod
    def split(self):
        pass

class TTSSplitter(BaseSplitter):
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        super().__init__(X, Y, SMILES, mols,
                 ID, type, test_split_r=test_split_r)

    def split(self):
        split = train_test_split(self.X, self.Y,
                                 test_size=self.test_split_r)
        X_train, X_test, Y_train, Y_test = split
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.Y_train = np.array(Y_train)
        self.Y_test = np.array(Y_test)


class BSplitter(BaseSplitter):
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        super().__init__(X, Y, SMILES, mols,
                 ID, type, test_split_r=test_split_r)

    def split(self):

        n_samples_test = int(self.Y.shape[0] * self.test_split_r)

        dataset_w_mol_obj = pd.DataFrame({'X': self.X, 'Y': self.Y,
                                          'mols': self.mols, 'ID': self.ID},
                                         index=self.ID)
        dataset_w_mol_obj['cluster'] = butina_cluster(dataset_w_mol_obj['mols'])
        uniq_cluster_ids = dataset_w_mol_obj['cluster'].value_counts().loc[lambda x: x == 1].index.tolist()
        if len(uniq_cluster_ids) < int(dataset_w_mol_obj.shape[0] * self.test_split_r):
            print('Unable to split dataset based on butina clustering')
        test_set = dataset_w_mol_obj[dataset_w_mol_obj.cluster.isin(uniq_cluster_ids)].sample(n_samples_test)
        train_set = dataset_w_mol_obj[~dataset_w_mol_obj['ID'].isin(test_set['ID'].tolist())]
        data = [train_set['X'], test_set['X'], train_set['Y'], test_set['Y']]
        data = [np.array(x.tolist()) for x in data]
        X_train, X_test, Y_train, Y_test = data
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test


class SSplitter(BaseSplitter):
    def __init__(self, X, Y, SMILES, mols,
                 ID, type, test_split_r=0.3):
        super().__init__(X, Y, SMILES, mols,
                 ID, type, test_split_r=test_split_r)

    def split(self):
        dataset_w_mol_obj = pd.DataFrame({'X': self.X, 'Y': self.Y,
                                          'Smiles String': self.SMILES,
                                          'mols': self.mols, 'ID': self.ID},
                                           index=self.ID)
        scaffold_sets = generate_scaffolds(dataset_w_mol_obj)
        train_cutoff = (1 - self.test_split_r) * self.Y.shape[0]
        train_inds = []
        test_inds = []
        for scaffold_set in scaffold_sets:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                test_inds += scaffold_set
            else:
                train_inds += scaffold_set
        X_train = dataset_w_mol_obj.iloc[train_inds]['X']
        X_test = dataset_w_mol_obj.iloc[test_inds]['X']
        Y_train = dataset_w_mol_obj.iloc[train_inds]['Y']
        Y_test = dataset_w_mol_obj.iloc[test_inds]['Y']
        data = [np.array(x.tolist()) for x in [X_train, X_test, Y_train, Y_test]]
        X_train, X_test, Y_train, Y_test = data
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train =Y_train
        self.Y_test = Y_test


SPLIT = {'TTS': TTSSplitter,
         'SS': SSplitter,
         'B': BSplitter}


class Validation:
    def __init__(self, model, X, Y, name):
        self.model = model
        self.X = X
        self.Y = Y
        self.name = name
        self.results = None
        self.run()

    @staticmethod
    def f_one_mcc_score(model, X_test, Y_test):
        y_pred = model.predict(X_test)
        f_one = f1_score(Y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        return f_one, (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    def validate(self, model, X_test, Y_test):
        test_predicted = model.predict_proba(X_test)
        auc, (lb_auc, ub_auc) = calc_auc_ci(Y_test, test_predicted[:, 1])
        f_one, mcc = self.f_one_mcc_score(model, X_test, Y_test)
        accuracy = accuracy_score(Y_test, self.model.predict(X_test))
        performance_stats = [lb_auc, auc, ub_auc, accuracy, f_one, mcc]
        return performance_stats

    def run(self):
        test_stats = self.validate(self.model, self.X, self.Y)
        results = pd.DataFrame([test_stats], index=[self.name],
                                    columns=['AUC lower estimate', 'AUC',
                                             'AUC upper estimate', 'accuracy',
                                             'F1', 'MCC'])
        self.results= results

class Model(ABC):
    def __init__(self, X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.test_performance = None
        self.validation_performance = None

    def run(self):
        self.train()
        self.validation()

    def train(self):
        pass

    def validation(self):
        pass



class DeepSCAMsModel(Model):

    def __init__(self, X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation):
        super().__init__(X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation)
        self.scaler2 = MinMaxScaler()
        self.Deep_SCAMS_model = None
        super().run()

    @staticmethod
    def format_for_DeepSCAMs(fitted_scaler, X):
        X_fitted = fitted_scaler.transform(X)
        return X_fitted

    def train(self):
        seed = 1234
        MLP = MLPClassifier(activation='tanh', alpha=0.0001, batch_size='auto', beta_1=0.9,
                            beta_2=0.999, early_stopping=False, epsilon=1e-08,
                            hidden_layer_sizes=(100, 1000, 1000), learning_rate='constant',
                            learning_rate_init=0.001, max_iter=200, momentum=0.9,
                            n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
                            random_state=1234, shuffle=True, solver='sgd', tol=0.0001,
                            validation_fraction=0.1, verbose=False, warm_start=False)

        self.scaler2 = self.scaler2.fit(self.X_train)
        X_train = self.format_for_DeepSCAMs(self.scaler2, self.X_train)
        self.Deep_SCAMS_model = MLP.fit(X_train, self.Y_train.T)

    def validation(self):
        X_test = self.format_for_DeepSCAMs(self.scaler2, self.X_test)
        test_performance = Validation(self.Deep_SCAMS_model, X_test, self.Y_test, 'Test').results
        self.test_performance = test_performance
        X_validation = self.format_for_DeepSCAMs(self.scaler2, self.X_validation)
        validation_performance  = Validation(self.Deep_SCAMS_model, X_validation,
                                             self.Y_validation, 'Validation').results
        self.validation_performance = validation_performance

def create_keras_model(shape=2256,
                       dropout=0.2):
    model = Sequential()
    model.add(Dense(800, input_dim=shape,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=["accuracy"])
    return model

class TensorFlowModel(Model):

    def __init__(self, X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation):
        super().__init__(X_train, Y_train,
                         X_test, Y_test, X_validation,
                         Y_validation)
        self.TFModel = KerasClassifier(create_keras_model)
        self.scaler = RobustScaler(quantile_range=(25, 75))
        super().run()

    @staticmethod
    def format_for_model(fitted_scaler, X):
        X_fitted = fitted_scaler.transform(X)
        return X_fitted

    def train(self):
        self.scaler = self.scaler.fit(self.X_train)
        X_train = self.format_for_model(self.scaler, self.X_train)
        self.TFModel.fit(X_train, self.Y_train.T)

    def validation(self):
        X_test = self.format_for_model(self.scaler, self.X_test)
        test_performance = Validation(self.TFModel, X_test,
                                      self.Y_test, 'Test').results
        self.test_performance = test_performance
        X_validation = self.format_for_model(self.scaler, self.X_validation)
        validation_performance = Validation(self.TFModel,
                                            X_validation, self.Y_validation, 'Validation').results
        self.validation_performance = validation_performance

class PipelineSW(Pipeline):
    def fit(self, X, y, sample_weight=None):
        """Fit and pass sample weights only to the last step"""
        if sample_weight is not None:
            kwargs = {self.steps[-1][0] + '__sample_weight': sample_weight}
        else:
            kwargs = {}
        return super().fit(X, y, **kwargs)



class ALModel(Model):
    def __init__(self, X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation, n_queries,
                 q_strategy,
                 iteration, n_initial=10):
        super().__init__(X_train, Y_train,
                 X_test, Y_test, X_validation,
                 Y_validation)
        self.n_initial = n_initial
        self.n_queries = n_queries
        self.model = KerasClassifier(create_keras_model)
        self.scaler = RobustScaler(quantile_range=(25, 75))
        self.q_strategy = q_strategy
        self.iteration = iteration
        self.pipe = PipelineSW([('scaler', self.scaler),
                                ('tf_mlp', self.model)])
        self.initialize_al()
        self.final_model = None
        self.run()

    @staticmethod
    def random_choise(X_train, n_initial):
        initial_idx = np.random.choice(range(len(X_train)),
                                       size=n_initial, replace=False)
        return initial_idx

    @staticmethod
    def gmen_no_nan(list_with_stats):
        g_v = gmean(list_with_stats)
        if not np.isnan(g_v):
            return g_v
        else:
            return 0

    def initialize_al(self):

        initial_idx = self.random_choise(self.X_train, self.n_initial)

        while len(set(self.Y_train[initial_idx])) != 2:  # Check if both classes are presented
            initial_idx = self.random_choise(self.X_train, self.n_initial)

        X, Y = self.X_train[initial_idx], self.Y_train[initial_idx]
        X_initial, y_initial = self.X_train[initial_idx], self.Y_train[initial_idx]
        X_pool, y_pool = np.delete(self.X_train, initial_idx, axis=0), \
                         np.delete(self.Y_train, initial_idx, axis=0)

        learner = ActiveLearner(
            estimator=self.pipe,
            query_strategy=self.q_strategy,
            X_training=X_initial, y_training=y_initial
        )


        # Calculate initial performance metrics on the test set
        performance_test = Validation(learner, self.X_test,
                                      self.Y_test, 'Test').results
        performance_test_l = [performance_test.iloc[0].tolist()]
        gmean_test = [self.gmen_no_nan(performance_test_l[0])]

        # Calculate initial performance metrics on the validation set
        performance_validation = Validation(learner, self.X_validation,
                                            self.Y_validation, 'Validation').results
        performance_validation_l = [performance_validation.iloc[0].tolist()]
        gmean_validation = [self.gmen_no_nan(performance_validation_l[0])]

        initial_data = [X, Y]
        pool = [X_pool, y_pool]

        return initial_data, pool, performance_test_l, performance_validation_l, \
               learner, [gmean_test, gmean_validation]

    def run(self):

        initial_data, pool, performance_test_l,\
        performance_validation_l, learner, gmeans = self.initialize_al()

        X, Y = initial_data
        X_pool, y_pool = pool
        gmean_test, gmean_validation = gmeans

        for i in range(self.n_queries - 1):
            query_idx, query_inst = learner.query(X_pool)
            ### Edit ###
            learner.teach(X_pool[query_idx], y_pool[query_idx])
            #learner.teach(X_pool[query_idx], y_pool[query_idx],
            #              epochs=50, batch_size=128, verbose=0)
            X = np.append(X, X_pool[query_idx], axis=0)
            Y = np.append(Y, y_pool[query_idx])
            X_pool, y_pool = np.delete(X_pool, query_idx, axis=0), \
                             np.delete(y_pool, query_idx, axis=0)

            performance_t_iter = Validation(learner, self.X_test,
                                            self.Y_test, 'Test').results
            performance_t_iter_l = performance_t_iter.iloc[0].tolist()
            gmean_test.append(self.gmen_no_nan(performance_t_iter_l))
            performance_test_l.append(performance_t_iter_l)
            ### Edit ###
            print(performance_test_l)

            performance_v_iter = Validation(learner, self.X_validation,
                                            self.Y_validation, 'Validation').results
            performance_v_iter_l = performance_v_iter.iloc[0].tolist()
            ### Edit ###
            #learner.estimator[1].model.save(Path('/home/khali/scams/Results/N_SP1_TTS') / 'TF_MLP_AL_{}.h5'.format(i+1))
            gmean_validation.append(self.gmen_no_nan(performance_v_iter_l))
            performance_validation_l.append(performance_v_iter_l)
        performance_test_l = np.array(performance_test_l)
        performance_test_l[np.isnan(performance_test_l)] = 0
        self.test_performance = list(np.max(np.array(performance_test_l), axis=0))

        self.test_performance = pd.DataFrame([self.test_performance], index=["test performance"],
                                              columns=['AUC lower estimate', 'AUC',
                                                       'AUC upper estimate', 'accuracy',
                                                       'F1', 'MCC'])
        performance_validation_l = np.array(performance_validation_l)
        performance_validation_l[np.isnan(performance_validation_l)] = 0
        self.validation_performance = list(np.max(np.array(performance_validation_l), axis=0))
        self.validation_performance = pd.DataFrame([self.validation_performance], index=["validation performance"],
                                              columns=['AUC lower estimate', 'AUC',
                                                       'AUC upper estimate', 'accuracy',
                                                       'F1', 'MCC'])
        integral_performace = np.concatenate((performance_test_l, performance_validation_l), axis=1)
        index_max_pef = np.argmax(gmean(integral_performace, axis=1))
        final_X, final_Y = X[0: index_max_pef + self.n_initial, ], Y[0: index_max_pef + self.n_initial, ]
        self.final_model = TensorFlowModel(final_X, final_Y, self.X_test,
                                           self.Y_test, self.X_validation, self.Y_validation).TFModel
        print('Argmax validation', np.argmax(np.array(gmean_validation)))
        print('Argmax test', np.argmax(np.array(gmean_test)))




if __name__ == "__main__":
    currnt_path = Path.cwd().parents[0] / 'Datasets'
    ap = argparse.ArgumentParser()
    ap.add_argument('-s_n', '--study_name', required=True,
                    help='Study name')
    ap.add_argument('-d_p', '--datasets_path', required=False,
                    default='{}'.format(currnt_path))
    args = ap.parse_args()
    instance = SCAMsPipeline(args.study_name, args.datasets_path)