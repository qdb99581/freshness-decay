import os
import random

import numpy as np
from scipy import stats
from scipy.io import loadmat
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer  # Preprocessing


class Config():
    # Configuration
    def __init__(self):
        # Data parameters
        self.data_root_path = 'D:/Repos/Python/freshness-decay/data/'
        self.derivative = True
        if self.derivative:
            self.selected_bands = [i for i in range(299)]
        else:
            self.selected_bands = [i for i in range(300)]
        self.regression = True
        self.save_path = "MLP_classifier.hdf5"
        self.mushroom_class = "B"
        self.train_ratio = 0.5  # 0.8 for NN, 0.5 for SVM.

        # Model parameters
        self.n_hidden_layers = 1
        self.activation = 'linear'

        # Training loop parameters
        self.n_KFold = 5
        self.n_epochs = 20000
        self.valid_ratio = 0.2  # valid = all_data * train_ratio * valid_ratio
        self.batch_size = 64
        self.learning_rate = 1e-4

        print(f"\nUsing class {self.mushroom_class} mushroom.")


def import_data(data_root_path, selected_bands, regression=False, derivative=True, mushroom_class="A", normalize=None, shuffle=True):
    """Import matlab matrix data into numpy array with labels

    Args:
        data_root_path (str): The path contains dates folders
        selected_bands (list of integer): A list of desired the bands
        regression (bool, optional): Import data for regression.
        derivative (bool, optional): Whether to use derivative data or not. Defaults to "True".
        mushroom_class (str, optional): Select class A or class B. Defaults to "A".
        normalize (str, optional): Select normalization method. Defaults to "None".
        shuffle (bool, optional)

    Returns:
        x_data (np array) [n_data, n_bands]: data with bands
        y_data (np array) [n_data]: data with labels
    """
    if regression:
        x_data, y_data = _load_regression_data(
            data_root_path, selected_bands, derivative, mushroom_class)
    else:
        x_data, y_data = _load_classification_data(
            data_root_path, selected_bands, derivative, mushroom_class)

    # Normalization
    if normalize == "zscore":
        print("Normalizing data by Z-Score...")
        x_data = stats.zscore(x_data, axis=1)

    # Shuffling
    if shuffle:
        shuffled_idx = [i for i in range(len(x_data))]
        random.shuffle(shuffled_idx)
        x_data, y_data = x_data[shuffled_idx, :], y_data[shuffled_idx]

    return x_data, y_data


def _load_regression_data(data_root_path, selected_bands, derivative, mushroom_class):
    dates = ['20200929', '20201027']

    mushroom_class = mushroom_class.upper()

    print("Importing data...")
    for folder_idx in tqdm(range(0, 30, 28)):
        if derivative:
            img_folder = mushroom_class + str(folder_idx) + '_DIFF/'
        else:
            img_folder = mushroom_class + str(folder_idx) + '_NO_DIFF/'

        cur_folder_path = data_root_path + \
            dates[folder_idx//28] + '/' + img_folder

        data_list = os.listdir(cur_folder_path)

        cur_data, cur_label = [], []
        label = folder_idx // 28

        for img in data_list:
            img_path = cur_folder_path + img
            img_mat = loadmat(img_path)
            d_mat_data = img_mat['data_out']
            data = np.array(d_mat_data)
            cur_data.append(data)
            cur_label.append(label)

        cur_data = np.array(cur_data).reshape(-1, len(selected_bands))
        cur_data = cur_data[:, selected_bands]
        cur_label = np.array(cur_label).reshape(len(cur_label))

        if folder_idx != 0:
            x_data, y_data = np.concatenate(
                (x_data, cur_data)), np.concatenate((y_data, cur_label))
        else:
            x_data, y_data = cur_data, cur_label

    return x_data, y_data


def _load_classification_data(data_root_path, selected_bands, derivative, mushroom_class):
    dates = ['20200929', '20201001', '20201003', '20201005', '20201007',
             '20201009', '20201011', '20201013', '20201015', '20201017',
             '20201019', '20201021', '20201023', '20201025', '20201027']

    mushroom_class = mushroom_class.upper()

    print("Importing data...")
    for folder_idx in tqdm(range(0, 30, 2)):
        if derivative:
            img_folder = mushroom_class + str(folder_idx) + '_DIFF/'
        else:
            img_folder = mushroom_class + str(folder_idx) + '_NO_DIFF/'

        cur_folder_path = data_root_path + \
            dates[folder_idx//2] + '/' + img_folder

        data_list = os.listdir(cur_folder_path)

        cur_data, cur_label = [], []
        label = folder_idx // 2

        for img in data_list:
            img_path = cur_folder_path + img
            img_mat = loadmat(img_path)
            d_mat_data = img_mat['data_out']
            data = np.array(d_mat_data)
            cur_data.append(data)
            cur_label.append(label)

        cur_data = np.array(cur_data).reshape(-1, len(selected_bands))
        cur_data = cur_data[:, selected_bands]
        cur_label = np.array(cur_label).reshape(len(cur_label))

        if folder_idx != 0:
            x_data, y_data = np.concatenate(
                (x_data, cur_data)), np.concatenate((y_data, cur_label))
        else:
            x_data, y_data = cur_data, cur_label
    return x_data, y_data


def train_test_split(x_data, y_data, train_ratio=0.8):
    """Split data into training and testing based on the train ratio

    Args:
        x_data (np array) [n_data, n_bands]: data with bands
        y_data (np array) [n_data]: data with labels
        train_ratio (int, optional): Defaults to 0.8.
        shuffle (bool, optional): Defaults to True.

    Returns:
        x_train: [n_train, n_bands]
        y_train: [n_train]
        x_test: [n_test]
        y_test: [n_test]
    """
    n_train = round(len(x_data) * train_ratio)
    x_train, y_train = x_data[:n_train, :], y_data[:n_train]
    x_test, y_test = x_data[n_train:, :], y_data[n_train:]

    return x_train, y_train, x_test, y_test


# Unused since sklearn's SVM does not accept one-hot encoding form.
def one_hot_encoding(y):
    return LabelBinarizer().fit_transform(y)


def plot_loss_history(hist):
    """Plot loss history produced by tensorflow

    Args:
        history (tensorflow history class): This is the output of model.fit
    """
    plt.plot(hist.history['loss'])
    plt.title('model loss history')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.show()


if __name__ == "__main__":
    opt = Config()
    x_data, y_data = import_data(
        'D:/Repos/Python/freshness-decay/data/',
        selected_bands=opt.selected_bands,
        regression=True,
        derivative=False,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=True
    )

    x_train, y_train, x_test, y_test = train_test_split(
        x_data, y_data, train_ratio=opt.train_ratio)

    print("Test")
