from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,Activation,TimeDistributed,RepeatVector
import random
import pandas as pd
from keras.models import load_model

path="./results_0_1_pred_class_A.xlsx"
all_0_6 = pd.read_excel(path)
all_0_6 = all_0_6.to_numpy()
all_0_6 = all_0_6.transpose()
path="./results_2_15_pred_class_A.xlsx"
all_7_15 = pd.read_excel(path)
all_7_15 = all_7_15.to_numpy()
all_7_15 = all_7_15.transpose()

train_number = 2
train_label_number = 13

index = [i for i in range(40)]
random.shuffle(index)
all_training_data = all_0_6[0:40,:]
all_training_data = all_training_data.reshape((40,train_number,1))
all_training_data = all_training_data[index,:,:]

all_training_label = all_7_15[0:40,:]
all_training_label = all_training_label.reshape((40,train_label_number,1))
all_training_label = all_training_label[index,:,:]

all_testing_data = all_0_6[40:50,:]
all_testing_data = all_testing_data.reshape((10,train_number,1))

all_testing_label = all_7_15[40:50,:]
all_testing_label = all_testing_label.reshape((10,train_label_number,1))

print(all_training_data.shape)
print(all_training_label.shape)

n_batch = 50
n_epoch = 10000
time_step = train_number
# create LSTM
model = Sequential()
model.add(LSTM(600, input_shape=(time_step, 1),unroll=True))
model.add(RepeatVector(train_label_number))
model.add(LSTM(550, return_sequences=True , unroll=True))
model.add(LSTM(500, return_sequences=True , unroll=True))
model.add(LSTM(450, return_sequences=True , unroll=True))
model.add(LSTM(400, return_sequences=True , unroll=True))
model.add(LSTM(350, return_sequences=True , unroll=True))
model.add(LSTM(300, return_sequences=True , unroll=True))

model.add(TimeDistributed(Dense(600,activation='relu')))
model.add(TimeDistributed(Dense(550,activation='relu')))
model.add(TimeDistributed(Dense(500,activation='relu')))
model.add(TimeDistributed(Dense(450,activation='relu')))
model.add(TimeDistributed(Dense(400,activation='relu')))
model.add(TimeDistributed(Dense(350,activation='relu')))
model.add(TimeDistributed(Dense(1,activation='linear')))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)
print(model.summary())

# train LSTM
model.fit(all_training_data, all_training_label, epochs=n_epoch, batch_size=n_batch, verbose=2)
model.save('LSTM_model')


# evaluate
result = model.predict(all_testing_data)
result = result.reshape((10,train_label_number))
result = result.transpose()
DF = pd.DataFrame(result)
save_excel_name = 'trand_result_ans_A.xlsx'
DF.to_excel(save_excel_name, index=False)
print('DF save FINISH')

# # testing part 2 (55)
#
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\all_0_2_result.xlsx'
# all_0_2 = pd.read_excel(path)
# all_0_2 = all_0_2.to_numpy()
# all_0_2 = all_0_2.transpose()
# all_0_2 = all_0_2.reshape(55,2,1)
# path='C:\\Users\\Lab219-1\\PycharmProjects\\pythonProject\\all_m_a_result.xlsx'
# all_m_a = pd.read_excel(path)
# all_m_a = all_m_a.to_numpy()
# all_m_a = all_m_a.transpose()
# all_m_a = all_m_a.reshape(55,2,1)
#
# result = model.predict(all_0_2)
# result = result.reshape((55,train_label_number))
# result = result.transpose()
# DF = pd.DataFrame(result)
# save_excel_name = 'test_ans_B02.xlsx'
# DF.to_excel(save_excel_name, index=False)
# print('DF save FINISH')
#
# result = model.predict(all_m_a)
# result = result.reshape((55,train_label_number))
# result = result.transpose()
# DF = pd.DataFrame(result)
# save_excel_name = 'test_ans_Ama.xlsx'
# DF.to_excel(save_excel_name, index=False)
# print('DF save FINISH')