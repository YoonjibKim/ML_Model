import numpy as np
from keras import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.utils import to_categorical
from matplotlib import pyplot as plt


class DNN:
    @classmethod
    def dnn_run(cls, training_data_array, training_label_array, testing_data_array, testing_label_array):
        X_tn = training_data_array
        y_tn = to_categorical(training_label_array)
        X_te = testing_data_array
        y_te = to_categorical(testing_label_array)

        n_feat = X_tn.shape[1]
        epo = 50
        n_class = 2

        model = Sequential()
        model.add(Dense(20, input_dim=n_feat))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(n_class))
        model.add(Activation('softmax'))

        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        hist = model.fit(X_tn, y_tn, epochs=epo, batch_size=5)

        print(model.evaluate(X_te, y_te)[1])

        epoch = np.arange(1, epo + 1)
        accuracy = hist.history['accuracy']
        loss = hist.history['loss']

        plt.plot(epoch, accuracy, label='accuracy')
        plt.plot(epoch, loss, label='loss')
        plt.xlabel('epoch')
        plt.ylabel('accuracy & loss')
        plt.legend()
        plt.savefig('Output_results/dnn_accuracy_loss.png')