import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential

from functions import data


def getData(set):
    if set == 'dummy':
        # Generate dummy data
        x_train = np.random.random((1000, 20))
        y_train = np.random.randint(2, size=(1000, 1))
        x_test = np.random.random((100, 20))
        y_test = np.random.randint(2, size=(100, 1))
        return x_train, y_train, x_test, y_test
    else:
        x_train, y_train, x_test, y_test = data.getData()
        x_train = np.reshape(x_train, (x_train.shape[0], -1))
        x_test = np.reshape(x_test, (x_test.shape[0], -1))
        return x_train, y_train, x_test, y_test


def visualize(model):
    from keras.utils import plot_model
    plot_model(model, to_file='model.png', show_shapes=True)


def plainBinaryClassifier():
    x_train, y_train, x_test, y_test = getData('dummy')
    # x_train, y_train, x_test, y_test = getData('real
    print(y_train.dtype)
    print(y_train[1])

    print('Training data shape: ', x_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

    input_dim = x_train.shape[1]

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=10,
              batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)
    visualize(model)


def dataTest():
    #  load the data

    x_train, y_train, x_test, y_test = data.getData()

    print('Training data shape: ', x_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)
    y_test_g = np.random.randint(2, size=(100, 1))

    # confirm shapes
    print('Training data shape: ', x_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ',x_test.shape)
    print('Test labels shape: ', y_test.shape)

def fcn():
    from keras.models import load_model
    best_model = load_model('best_fcn_model.h5')


fcn()
# dataTest()
# plainBinaryClassifier()
