from abc import ABC
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.neural_network import MLPClassifier
from validation import Validation

#  Importing modules from TensorFlow and Keras
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout

#  Save model
import h5py
import io
import copy


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
        validation_performance = Validation(self.Deep_SCAMS_model, X_validation,
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
