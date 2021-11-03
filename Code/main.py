#  Imports
from __future__ import print_function

#  Importing general-purpose modules
from pathlib import Path
import pandas as pd
import numpy as np

import argparse
import time


#  Importing modules from sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

#  Importing modules from scipy
from scipy import stats
from scipy.signal import savgol_filter
from scipy.stats import gmean

#  Importing the modules from imbalanced learning
from imblearn.over_sampling import ADASYN, SMOTE  # up-sampling
from imblearn.under_sampling import CondensedNearestNeighbour, InstanceHardnessThreshold  # down-sampling

#  Importing the modules for active learning
from modAL.uncertainty import uncertainty_sampling, entropy_sampling


#  Importing modules to filter out warnings
from rdkit import RDLogger
import warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")


from dataset import Dataset
from utilities import dataset_to_splitter, str2bool, rm_tree, file_doesnot_exist, sampl
from splitters import BaseSplitter, TTSSplitter, BSplitter, SSplitter
from models import DeepSCAMsModel, TensorFlowModel,  ALModel

#  Defining variables to automatically run the pipeline

#  Up- and down-sampling
SAMPLING = {'N': False,
            'SMOTE': SMOTE(),
            'ADASYN': ADASYN(),
            'CNN': CondensedNearestNeighbour(),
            'IHT': InstanceHardnessThreshold()}

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
                 ID_name='ID', iterations=1,
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


SPLIT = {'TTS': TTSSplitter,
         'SS': SSplitter,
         'B': BSplitter}

# class PipelineSW(Pipeline):
#     def fit(self, X, y, sample_weight=None):
#         """Fit and pass sample weights only to the last step"""
#         if sample_weight is not None:
#             kwargs = {self.steps[-1][0] + '__sample_weight': sample_weight}
#         else:
#             kwargs = {}
#         return super().fit(X, y, **kwargs)

if __name__ == "__main__":
    currnt_path = Path.cwd().parents[0] / 'Datasets'
    ap = argparse.ArgumentParser()
    ap.add_argument('-s_n', '--study_name', required=True,
                    help='Study name')
    ap.add_argument('-d_p', '--datasets_path', required=False,
                    default='{}'.format(currnt_path))
    args = ap.parse_args()
    instance = SCAMsPipeline(args.study_name, args.datasets_path)