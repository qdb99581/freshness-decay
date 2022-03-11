from random import shuffle
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, RepeatVector


class Configs:
    def __init__(self):
        print(
            f"There are {len(tf.config.experimental.list_physical_devices('GPU'))} GPUs Available: ")

    n_train_labels = 13

    n_batch = 50
    n_epoch = 10000
    timesteps = 2
    learning_rate = 0.001


def my_load_excel(filepath):
    data = pd.read_excel(filepath)
    data = data.to_numpy()
    data = data.transpose()

    return data


def my_train_test_split(data_d0_d6, data_d7_d15, test_size=0.2, n_train_days=2, pred_days=13):
    n_train_data = int(50 * (1-test_size))
    n_test_data = int(50 - n_train_data)

    print(
        f"Split dataset into {n_train_data} training data and {n_test_data} testing data.")

    shuffled_index = [i for i in range(n_train_data)]
    shuffle(shuffled_index)

    train_X = data_d0_d6[:n_train_data, :]
    train_X = train_X.reshape((n_train_data, n_train_days, 1))
    train_X = train_X[shuffled_index, :, :]

    train_y = data_d7_d15[:n_train_data, :]
    train_y = train_y.reshape((n_train_data, pred_days, 1))
    train_y = train_y[shuffled_index, :, :]

    test_X = data_d0_d6[n_train_data:, :]
    test_X = test_X.reshape((n_test_data, n_train_days, 1))

    test_y = data_d7_d15[n_train_data:, :]
    test_y = test_y.reshape((10, pred_days, 1))

    return train_X, train_y, test_X, test_y


def plot_loss_history(hist):
    """Plot loss history

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
    opt = Configs()

    data_d0_d6 = my_load_excel("./results_0_1_pred_class_A.xlsx")
    data_d7_d15 = my_load_excel("./results_2_15_pred_class_A.xlsx")

    train_X, train_y, test_X, test_y = my_train_test_split(
        data_d0_d6, data_d7_d15)

    # Construct model
    model = Sequential()
    model.add(LSTM(600, input_shape=(opt.timesteps, 1), unroll=True))
    model.add(RepeatVector(opt.n_train_labels))
    model.add(LSTM(550, return_sequences=True, unroll=True))
    model.add(LSTM(500, return_sequences=True, unroll=True))
    model.add(LSTM(450, return_sequences=True, unroll=True))
    model.add(LSTM(400, return_sequences=True, unroll=True))
    model.add(LSTM(350, return_sequences=True, unroll=True))
    model.add(LSTM(300, return_sequences=True, unroll=True))

    model.add(TimeDistributed(Dense(600, activation='relu')))
    model.add(TimeDistributed(Dense(550, activation='relu')))
    model.add(TimeDistributed(Dense(500, activation='relu')))
    model.add(TimeDistributed(Dense(450, activation='relu')))
    model.add(TimeDistributed(Dense(400, activation='relu')))
    model.add(TimeDistributed(Dense(350, activation='relu')))
    model.add(TimeDistributed(Dense(1, activation='linear')))

    adam = keras.optimizers.Adam(learning_rate=opt.learning_rate)
    model.compile(loss='mean_squared_error', optimizer=adam)
    # print(model.summary())

    history = model.fit(train_X, train_y, epochs=opt.n_epoch, batch_size=opt.n_batch,
                        verbose='auto')

    plot_loss_history(history)
    model.save('LSTM_model')

    # Evaluate
    n_test_data = 10
    result = model.predict(test_X)
    result = result.reshape((n_test_data, opt.n_train_labels))
    result = result.transpose()
    df = pd.DataFrame(result)
    save_excel_name = 'decay_trend_pred_class_A.xlsx'
    df.to_excel(save_excel_name, index=False)
    print("Decay trend prediction saved!")
