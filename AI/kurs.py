import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Установка путей к директориям
current_directory = os.getcwd()
TrainDir = os.path.join(current_directory, 'train')
TestDir = os.path.join(current_directory, 'test')

# Подготовка данных с аугментацией
Width = 150
Height = 150
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255)

TrainGenerator = train_datagen.flow_from_directory(
    TrainDir,
    target_size=(Width, Height),
    batch_size=batch_size,
    class_mode='categorical'  # Для многоклассовой классификации
)

TestGenerator = validation_datagen.flow_from_directory(
    TestDir,
    target_size=(Width, Height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Важно, чтобы порядок оставался прежним
)

# Архитектура нейронной сети
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(Width, Height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(TrainGenerator.class_indices), activation='softmax'))

# Оптимизатор
optimizer = Adam(learning_rate=0.001)

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Обучение модели
model.fit(TrainGenerator, epochs=5, validation_data=TestGenerator)

# Оценка модели
loss, accuracy = model.evaluate(TestGenerator)
print(f'Test accuracy: {accuracy * 100:.2f}%')

# Получение предсказаний
predictions = model.predict(TestGenerator)

# Получение названий классов
class_labels = list(TestGenerator.class_indices.keys())

# Визуализация предсказаний
num_images_to_show = 10  # Количество изображений для отображения
plt.figure(figsize=(15, 10))

for i in range(num_images_to_show):
    plt.subplot(2, 5, i + 1)

    # Отображение изображения
    img = plt.imread(os.path.join(TestDir, TestGenerator.filenames[i]))
    plt.imshow(img)
    plt.axis('off')

    # Обработка предсказаний
    predicted_class_index = np.argmax(predictions[i])  # Индекс класса с наибольшей вероятностью
    predicted_breed = class_labels[predicted_class_index]  # Получение названия породы
    probability = predictions[i][predicted_class_index]  # Вероятность предсказания

    plt.title(f'Predicted: {predicted_breed}\nProb: {probability:.2f}')

plt.tight_layout()
plt.show()
