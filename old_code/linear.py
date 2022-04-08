import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import scipy.io
import os
import keras
import random
from joblib import dump
import pandas as pd


def loading_data(path, train_length, test_length, bands):
    # LOADING DATA
    data_list = os.listdir(path)
    data_1 = []
    for i in range(len(data_list)):
        sigal_path = path+data_list[i]
        d_mat = scipy.io.loadmat(sigal_path)
        d_mat_data = d_mat['data_out']
        data = np.array(d_mat_data)
        data_1.append(data)
    data_1 = np.array(data_1).reshape(-1, np.array(data_1).shape[2])
    traing_data = data_1[0:(0+train_length), bands]
    testing_data = data_1[(0+train_length):(0+train_length+test_length), bands]
    return traing_data, testing_data


def build_label():
    training_label_0 = [0 for i in range(40)]
    training_label_28 = [1 for i in range(40)]

    testing_label_0 = [0 for i in range(10)]
    testing_label_28 = [1 for i in range(10)]

    all_training_label = training_label_0+training_label_28
    all_testing_label = testing_label_0+testing_label_28
    return all_training_label, all_testing_label


# band_selection = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,236,237,238,239,242,243,244,245,246,247,248,249]
# band_selection = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298]
band_selection = [i for i in range(299)]
bands_num = len(band_selection)
# print('A 級測試 ')
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\A0\\'
# traing_data_A0,testing_data_A0 = LOADING_DATA(path,40,10,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\A28\\'
# traing_data_A28,testing_data_A28 = LOADING_DATA(path,40,10,band_selection)


print('B 級測試 ')
path = 'C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B0\\'
traing_data_A0, testing_data_A0 = loading_data(path, 40, 10, band_selection)
path = 'C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B2\\'
traing_data_A2, testing_data_A2 = loading_data(path, 40, 10, band_selection)
path = 'C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B28\\'
traing_data_A28, testing_data_A28 = loading_data(path, 40, 10, band_selection)
traing_data_A0 = traing_data_A0[0:20, :]
traing_data_A2 = traing_data_A2[0:20, :]


all_training_data = np.concatenate(
    (traing_data_A0, traing_data_A2, traing_data_A28))
all_testing_data = np.concatenate((testing_data_A0, testing_data_A28))

# make training and testing label
all_training_label, all_testing_label = build_label()
all_training_label = np.array(all_training_label)
all_testing_label = np.array(all_testing_label)

shuffle_index = [i for i in range(all_training_data.shape[0])]
random.shuffle(shuffle_index)
all_training_data = all_training_data[shuffle_index, :]
all_training_label = all_training_label[shuffle_index]

# all_training_label_onehot =np_utils.to_categorical(all_training_label) #training one hot
# all_testing_label_onehot =np_utils.to_categorical(all_testing_label)  #testing one hot


# DNN model
model = Sequential()
model.add(Dense(256, activation='relu'))
# model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(Dense(256, activation='relu'))
# model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(Dense(256, activation='relu'))
# model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
# model.add(keras.layers.LeakyReLU(alpha=0.2))
model.add(Dropout(0.3))
model.add(Dense(1, activation='linear'))
model.build(input_shape=(None, bands_num))
model.summary()
adam = keras.optimizers.adam(lr=0.0001)
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['accuracy'])
model.fit(all_training_data, all_training_label, batch_size=80,
          validation_split=0.25, epochs=10000, verbose=2)
# weight = model.get_weights()


model.save('DNN_relu_linear_model')
# model = keras.models.load_model('DNN_model')
testing_score = model.evaluate(all_testing_data, all_testing_label)
print('testing_loss = ', testing_score[0],
      '   |  ', 'testing_ACC = ', testing_score[1])

#
#
# # testing data prepare
# def LOADING_test_DATA(path,bands):
#     # LOADING DATA
#     data_list=os.listdir(path)
#     data_1=[]
#     for i in range(len(data_list)):
#         sigal_path=path+data_list[i]
#         d_mat = scipy.io.loadmat(sigal_path)
#         d_mat_data = d_mat['data_out']
#         data= np.array(d_mat_data)
#         data_1.append(data)
#     data_1=np.array(data_1).reshape(-1,np.array(data_1).shape[2])
#     data_1=data_1[:,bands]
#     return data_1
#
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B0\\'
# testing_data_A0 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B2\\'
# testing_data_A2 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B4\\'
# testing_data_A4 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B6\\'
# testing_data_A6 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B8\\'
# testing_data_A8 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B10\\'
# testing_data_A10 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B12\\'
# testing_data_A12 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B14\\'
# testing_data_A14 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B16\\'
# testing_data_A16 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B18\\'
# testing_data_A18 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B20\\'
# testing_data_A20 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B22\\'
# testing_data_A22 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B24\\'
# testing_data_A24 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B26\\'
# testing_data_A26 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(1)\\B28\\'
# testing_data_A28 = LOADING_test_DATA(path,band_selection)
#
# all_testing_data_test=[]
# all_testing_data_test.append(testing_data_A0)
# all_testing_data_test.append(testing_data_A2)
# all_testing_data_test.append(testing_data_A4)
# all_testing_data_test.append(testing_data_A6)
# all_testing_data_test.append(testing_data_A8)
# all_testing_data_test.append(testing_data_A10)
# all_testing_data_test.append(testing_data_A12)
# all_testing_data_test.append(testing_data_A14)
# all_testing_data_test.append(testing_data_A16)
# all_testing_data_test.append(testing_data_A18)
# all_testing_data_test.append(testing_data_A20)
# all_testing_data_test.append(testing_data_A22)
# all_testing_data_test.append(testing_data_A24)
# all_testing_data_test.append(testing_data_A26)
# all_testing_data_test.append(testing_data_A28)
# all_testing_data_test=np.array(all_testing_data_test)
#
#
# all_0_1_result=[]
# all_2_15_result=[]
# for i in range(all_testing_data_test.shape[0]):
#     if i<2:
#         one_day = []
#         one_data = all_testing_data_test[i]
#         for j in range(50):
#             a = one_data[j,:].reshape((1,len(band_selection)))
#             ans = model.predict(a)
#             one_day.append(ans)
#         all_0_1_result.append(one_day)
#     else:
#         one_day = []
#         one_data = all_testing_data_test[i]
#         for j in range(50):
#             a = one_data[j, :].reshape((1, len(band_selection)))
#             ans = model.predict(a)
#             one_day.append(ans)
#         all_2_15_result.append(one_day)
#
# all_0_1_result=np.array(all_0_1_result).reshape((2,50))
# all_2_15_result=np.array(all_2_15_result).reshape((13,50))
# df0_1 = pd.DataFrame(all_0_1_result)
# df2_15 = pd.DataFrame(all_2_15_result)
# save_excel_name = 'all_0_1_result_ans(A).xlsx'
# df0_1.to_excel(save_excel_name, index=False)
# print('all_0_1_result_ans save FINISH')
# save_excel_name = 'all_2_15_result_ans(A).xlsx'
# df2_15.to_excel(save_excel_name, index=False)
# print('all_2_15_result_ans save FINISH')
#
#
# # Testing part2(55)
#
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(test)\\B0\\'
# testing_data_A0 = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(test)\\B1a\\'
# testing_data_A1a = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(test)\\B1m\\'
# testing_data_A1m = LOADING_test_DATA(path,band_selection)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\DIFF(test)\\B2\\'
# testing_data_A2 = LOADING_test_DATA(path,band_selection)
#
# testdata_A0A2=[]
# testdata_A1ma=[]
#
# testdata_A0A2.append(testing_data_A0)
# testdata_A0A2.append(testing_data_A2)
# testdata_A0A2=np.array(testdata_A0A2)
#
# testdata_A1ma.append(testing_data_A1m)
# testdata_A1ma.append(testing_data_A1a)
# testdata_A1ma=np.array(testdata_A1ma)
#
#
#
# all_0_2_result=[]
# all_m_a_result=[]
#
#
# for i in range(testdata_A0A2.shape[0]):
#     one_day_02 = []
#     one_day_ma = []
#     one_data_02 = testdata_A0A2[i]
#     one_data_ma = testdata_A1ma[i]
#     for j in range(55):
#         a = one_data_02[j, :].reshape((1, len(band_selection)))
#         b = one_data_02[j, :].reshape((1, len(band_selection)))
#         ans = model.predict(a)
#         ans2 = model.predict(b)
#         one_day_02.append(ans)
#         one_day_ma.append(ans2)
#     all_0_2_result.append(one_day_02)
#     all_m_a_result.append(one_day_ma)
#
# all_0_2_result=np.array(all_0_2_result).reshape((2,55))
# all_m_a_result=np.array(all_m_a_result).reshape((2,55))
# df0_2 = pd.DataFrame(all_0_2_result)
# dfm_a = pd.DataFrame(all_m_a_result)
# save_excel_name = 'all_0_2_result.xlsx'
# df0_2.to_excel(save_excel_name, index=False)
# print('all_0_2_result save FINISH')
# save_excel_name = 'all_m_a_result.xlsx'
# dfm_a.to_excel(save_excel_name, index=False)
# print('all_m_a_result save FINISH')
