import os
import random
import pickle

import numpy as np
from scipy import stats
from scipy.io import loadmat
import imageio
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer  # Preprocessing


class Config():
    # Configuration
    def __init__(self):
        # Data parameters
        self.data_root_path = '../data/'
        self.derivative = True
        if self.derivative:
            self.selected_bands = [i for i in range(299)]
        else:
            self.selected_bands = [i for i in range(300)]
        self.regression = True
        self.save_path = "./original_regr_A/cp-{epoch:04d}"
        # self.save_path = "MLP_regression_A_relu.hdf5"
        self.mushroom_class = "A"
        self.train_ratio = 0.8  # 0.8 for NN, 0.5 for SVM.

        self.mlp_layout = {
            'MLP21': [64, 32],
            'MLP22': [512, 512],
            'MLP31': [128, 64, 32],
            'MLP32': [512, 512, 512],
            'MLP41': [256, 128, 64, 32],
            'MLP42': [512, 512, 512, 512],
            'MLP51': [512, 256, 128, 64, 32],
            'MLP52': [512, 512, 512, 512, 512]
        }
        self.svm_layout = {
            'lSVM1': [0.01],
            'lSVM2': [1],
            'lSVM3': [100],
            'kSVM1': [0.01, 0.001],
            'kSVM2': [0.01, 'scale'],
            'kSVM3': [0.01, 'auto'],
            'kSVM4': [1, 0.001],
            'kSVM5': [1, 'scale'],
            'kSVM6': [1, 'auto'],
            'kSVM7': [100, 0.001],
            'kSVM8': [100, 'scale'],
            'kSVM9': [100, 'auto']
        }

        # Model parameters
        self.n_hidden_layers = 1
        self.activation = 'relu'

        # Training loop parameters
        self.n_KFold = 5
        self.n_epochs = 20000
        self.valid_ratio = 0.2  # valid = all_data * train_ratio * valid_ratio
        self.batch_size = 64
        self.learning_rate = 1e-4

        print(f"\nUsing class {self.mushroom_class} mushroom.")


def import_data(
    data_root_path,
    selected_bands,
    train_for_regression=False,
    derivative=True,
    mushroom_class="A",
    normalize=None,
    shuffle=True
):
    """Import matlab matrix data into numpy array with labels

    Args:
        data_root_path (str): The path contains dates folders
        selected_bands (list of integer): A list of desired the bands
        train_for_regression (bool, optional): Import data of day 0 and day 28 for training regression.
        derivative (bool, optional): Whether to use derivative data or not. Defaults to "True".
        mushroom_class (str, optional): Select class A or class B. Defaults to "A".
        normalize (str, optional): Select normalization method. Defaults to "None".
        shuffle (bool, optional)

    Returns:
        x_data (np array) [n_data, n_bands]: data with bands
        y_data (np array) [n_data]: data with labels
    """
    if train_for_regression:
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
    """Import data for regression. Train on the first day and last day only.

    Args:
        data_root_path (_type_): _description_
        selected_bands (_type_): _description_
        derivative (_type_): _description_
        mushroom_class (_type_): _description_

    Returns:
        _type_: _description_
    """
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
        label = folder_idx // 28  # Label with 0 and 1.

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


def make_gif(filedir, gif_name, duration=0.04):
    filenames = os.listdir(filedir)

    images = []
    for filename in tqdm(filenames):
        images.append(imageio.imread(filedir + filename))
    imageio.mimsave(filedir+"/"+gif_name+".gif", images, duration=duration)


def compute_scores(layout_acc_dict):
    """Compute the scores by given dictionary, which has the form: {'model_id': [score_1, score_2, score_3]}

    Args:
        layout_acc_dict (dict): Dictionary with every layout and its list of scores

    Returns:
        dict: A dictionary with every layout and its score +/- std.
    """
    score_dict = {}
    for layout_id, acc_list in layout_acc_dict.items():
        cur_mean = round(np.mean(acc_list) * 100, 2)
        cur_std = round(np.std(acc_list) * 100, 2)
        score = f"{cur_mean:2.2f}% +/- {cur_std:2.2f}%"

        score_dict[layout_id] = score

    return score_dict


def plot_double_bars(derivative_dict, reflectance_dict, mushroom_class):
    x = list(derivative_dict.keys())
    if x[0][0] == 'l' or x[0][0] == 'k':
        model = 'SVM'
    else:
        model = 'MLP'

    x_axis = np.arange(len(x))
    der_acc = list(derivative_dict.values())
    ref_acc = list(reflectance_dict.values())

    plt.figure(figsize=(10, 6))
    plt.bar(x_axis-0.2, der_acc, 0.4,
            label='Derivative Spectrum', edgecolor='black')
    plt.bar(x_axis+0.2, ref_acc, 0.4,
            label='Reflectance Spectrum', edgecolor='black')
    plt.xticks(x_axis, x)
    plt.xlabel("Model")
    plt.ylabel("Mean Accuracy")
    plt.ylim((0, 1.05))
    plt.title(
        f"The accuracy of freshness by {model} on class {mushroom_class}")
    plt.legend()
    plt.savefig(
        f"./mlp_svm_results/{model}_results_{mushroom_class}.png")    
    # plt.show()


def df2dict(report_df):
    id = report_df['Model ID']
    scores = report_df['Scores']

    scores_dict = {}
    for i in range(len(report_df.index)):
        cur_id = id.iloc[i]
        cur_score_str = scores.iloc[i]
        score = _get_score_from_str(cur_score_str)
        scores_dict[cur_id] = score

    return scores_dict


def _get_score_from_str(score_str):
    """This function extracts score from string. e.g., 9.87% +/- 2.01% as input
        will get 9.87 as float answer.
    Args:
        score_str (str): score as string.

    Returns:
        _float_: score
    """
    pointer = 0
    for idx, char in enumerate(score_str):
        if char == '%':
            pointer = idx
            break

    return float(score_str[:pointer]) / 100


if __name__ == "__main__":
    import pandas as pd

    opt = Config()

    model = 'svm'
    mushroom_class = "A"

    ref_df = pd.read_csv(f"./mlp_svm_results/{model}_results_{mushroom_class}_reflectance.csv")
    der_df = pd.read_csv(f"./mlp_svm_results/{model}_results_{mushroom_class}_derivative.csv")

    ref_dict = df2dict(ref_df)
    der_dict = df2dict(der_df)

    plot_double_bars(der_dict, ref_dict, f'{mushroom_class}')
