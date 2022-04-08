import os
from tqdm import tqdm
import pandas as pd
import random
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from scipy.io import loadmat


def load_data(data_path, num_train_data, num_test_data, selected_bands):
    """Load only desired number of data with desired bands

    Args:
        data_path (string)
        num_train_data (int)
        num_test_data (int)
        selected_bands (List): List of selected bands

    Returns:
        training_data(List), testing_data(List): training data and testing data with 
                             dimension of [num_data, num_bands]
    """
    data_list = os.listdir(data_path)
    num_data = len(data_list)
    res_data = []

    for i in range(num_data):
        img_path = data_path + data_list[i]
        d_mat = loadmat(img_path)
        d_mat_data = d_mat['data_out']
        data = np.array(d_mat_data)
        res_data.append(data)

    res_data = np.array(res_data).reshape(-1, np.array(res_data).shape[2])
    training_data = res_data[0:(0 + num_train_data), selected_bands]
    testing_data = res_data[(0 + num_train_data):(0 +
                                                  num_train_data + num_test_data), selected_bands]

    return training_data, testing_data


def build_label(num_train_data, num_test_data):
    """Create lists of labels

    Args:
        num_train_data (List)
        num_test_data (List)

    Returns:
        List, List: 
    """
    training_label_0 = [0 for i in range(num_train_data // 2)]
    training_label_28 = [1 for i in range(num_train_data // 2)]

    testing_label_0 = [0 for i in range(num_test_data // 2)]
    testing_label_28 = [1 for i in range(num_test_data // 2)]

    all_training_label = np.array(training_label_0 + training_label_28)
    all_testing_label = np.array(testing_label_0 + testing_label_28)

    return all_training_label, all_testing_label


num_bands = 299
seleted_band = [i for i in range(num_bands)]
num_train_data = 40
num_test_data = 10

# Load training data
print('Load class A data...')
path = 'D:\\Repos\\Python\\freshness-decay\\data\\20200929\\A0_DIFF\\'
training_data_A0, testing_data_A0 = load_data(
    path, num_train_data, num_test_data, seleted_band)
path = 'D:\\Repos\\Python\\freshness-decay\\data\\20201001\\A2_DIFF\\'
training_data_A2, testing_data_A2 = load_data(
    path, num_train_data, num_test_data, seleted_band)
path = 'D:\\Repos\\Python\\freshness-decay\\data\\20201027\\A28_DIFF\\'
training_data_A28, testing_data_A28 = load_data(
    path, num_train_data, num_test_data, seleted_band)

# 20 + 20 + 40
train_X = np.concatenate(
    (training_data_A0[:20, :], training_data_A2[:20, :], training_data_A28))
# 40 + 40
test_X = np.concatenate((testing_data_A0[:20, :], testing_data_A28))

print(f"Size of training data: {len(train_X)}")
print(f"Size of testing data: {len(test_X)}")

# Create labels
print("Creating labels...")
train_y, test_y = build_label(len(train_X), len(test_X))

# Shuffling
shuffle_index = [i for i in range(len(train_X))]
random.shuffle(shuffle_index)
train_X = train_X[shuffle_index, :]
train_y = train_y[shuffle_index]

# Construct DNN model
model = Sequential([
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')
])

model.build(input_shape=(None, num_bands))
model.summary()

adam = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])

model.fit(train_X, train_y, batch_size=80,
          validation_split=0.25, epochs=500, verbose=1)

model.save('DNN_relu_linear_model')

test_score = model.evaluate(test_X, test_y)
print('testing_loss = ', test_score[0],
      '   |  ', 'testing_ACC = ', test_score[1])

##############
# Evaluation #
##############

def load_eval_data(num_bands):
    date = ['20200929', '20201001', '20201003', '20201005', '20201007',
            '20201009', '20201011', '20201013', '20201015', '20201017',
            '20201019', '20201021', '20201023', '20201025', '20201027']

    test_data = []

    for i in tqdm(range(0, 30, 2)):
        folder_path = 'D:/Repos/Python/freshness-decay/data/' + date[i//2] + '/A' + \
            str(i) + '_DIFF/'

        data_list = os.listdir(folder_path)
        ret_data = []

        for j in range(len(data_list)):
            img_path = folder_path + data_list[j]
            d_mat = loadmat(img_path)
            d_mat_data = d_mat['data_out']
            data = np.array(d_mat_data)
            ret_data.append(data)

        ret_data = np.array(ret_data).reshape(-1, np.array(ret_data).shape[2])
        ret_data = ret_data[:, num_bands]

        test_data.append(ret_data)

    test_data = np.array(test_data)

    return test_data

test_data = load_eval_data(num_bands=seleted_band)

print("Evaluating...")
results_0_1 = []
results_2_15 = []

for i in tqdm(range(test_data.shape[0])):
    one_day = []
    cur_data = test_data[i]
    if i < 2:
        for j in range(50):
            a = cur_data[j, :].reshape((1, len(seleted_band)))
            pred = model.predict(a)
            one_day.append(pred)
        results_0_1.append(one_day)
    else:
        for j in range(50):
            a = cur_data[j, :].reshape((1, len(seleted_band)))
            pred = model.predict(a)
            one_day.append(pred)
        results_2_15.append(one_day)

results_0_1 = np.array(results_0_1).reshape((2, 50))
results_2_15 = np.array(results_2_15).reshape((13, 50))
df0_1 = pd.DataFrame(results_0_1)
df2_15 = pd.DataFrame(results_2_15)
save_excel_name = 'results_0_1_pred_class_A.xlsx'
df0_1.to_excel(save_excel_name, index=False)
save_excel_name = 'results_2_15_pred_class_A.xlsx'
df2_15.to_excel(save_excel_name, index=False)

print("Prediction saved!")