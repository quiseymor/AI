import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot as plt

(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
# X_train: изображения для обучения
# y_train: метки
# X_valid: изображения для проверки
# y_valid: метки 

print(X_train.shape)
print(X_valid.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_train[0:12])


plt.figure(figsize=(5,5))
for k in range(12):
    plt.subplot(3, 4, k+1)
    plt.imshow(X_train[k], cmap='Greys')
    plt.axis('off')
plt.tight_layout()
plt.show()

X_train = X_train.reshape(60000, 784).astype('float32')
X_valid = X_valid.reshape(10000, 784).astype('float32')
X_train /= 255 #chelochis -> vesh
X_valid /= 255
print(y_train[0])

n_classes = 10
y_train = keras.utils.to_categorical(y_train, n_classes)
y_valid = keras.utils.to_categorical(y_valid, n_classes)
print(y_train[0])

model = Sequential()
model.add(Dense(32, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))


model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01),
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=64, epochs=300, verbose=1, validation_data=(X_valid, y_valid))

test_digits = X_valid[0:10]
predictions = model.predict(test_digits)
print(predictions[0])
print(predictions[0].argmax())

