# Загрузка зависимостей:
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.applications.vgg19 import VGG19
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

def nn1(pooling: bool, epochs: int):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(inputs)
    if pooling:
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    if pooling:
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # вывести параметры
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))


def nn2(pooling: bool, epochs: int):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(filters=8, kernel_size=3, activation="relu")(inputs)
    if pooling:
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
   # if pooling:
    #    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    #if pooling:
     #   x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    if pooling:
        x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)

    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # вывести параметры
    model.summary()

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01), metrics=['accuracy'])

    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(X_valid, y_valid))


if __name__ == '__main__':
    #nn1(pooling=True, epochs=100)
    # nn1(pooling=True, epochs=200)
    # nn1(pooling=False, epochs=100)
    # nn1(pooling=False, epochs=200)
     #nn2(pooling=True, epochs=100)
     #nn2(pooling=True, epochs=200)
     nn2(pooling=False, epochs=100)
    #nn2(pooling=False, epochs=200)
