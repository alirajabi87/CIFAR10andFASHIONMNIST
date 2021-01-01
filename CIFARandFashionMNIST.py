"""
This file is part of my training, feel free to download it and use it
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense, ELU, Conv2D, MaxPool2D, Flatten, Input, GlobalMaxPool2D, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Fashion MNIST
from tensorflow.keras.datasets import cifar10, fashion_mnist

dataset = cifar10

(x_train, y_train), (x_test, y_test) = dataset.load_data()
x_train, x_test = x_train / 255., x_test / 255.

# k = len(set(y_train))

if dataset == fashion_mnist:
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

# if dataset == cifar10:
#     y_train, y_test = y_train.flatten(), y_test.flatten()


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print(x_train.shape)
print(x_train[0].max())
print(y_train_cat.shape)
# k = len(set(y_train))

k = y_train_cat.shape[1]
print(k)

Imageshape = x_train[0].shape
print(Imageshape)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
batchSize = 32
dataGenerator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
trainData = dataGenerator.flow(x_train, y_train_cat, batchSize)
StepPerTrain = trainData.n // batchSize

i = Input(shape=Imageshape)
# x = Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(i)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(x)
# x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(2, 2))(x)

# x = GlobalMaxPool2D()(x)
x = Flatten()(x)
x = Dropout(0.4)(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(k, activation='softmax')(x)

model = Model(i, x)

# model = Sequential([
#     Conv2D(32, (3, 3), input_shape=Imageshape, activation=ELU(0.1)),
#     MaxPool2D(pool_size=(2,2), strides=1),
#     Conv2D(64, (3, 3), activation=ELU(0.1)),
#     MaxPool2D(pool_size=(2,2), strides=1),
#     Conv2D(64, (3, 3), activation=ELU(0.1)),
#     MaxPool2D(pool_size=(2,2), strides=1),
#
#     Flatten(),
#
#     Dense(256, activation='relu'),
#     Dropout(0.5),
#     Dense(k, activation='softmax'),
# ])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
results = model.fit(trainData, #x_train, y_train_cat,
                    steps_per_epoch=StepPerTrain,
                    epochs=20,
                    validation_data=(x_test, y_test_cat), verbose=2)

plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.show()
model.save('CIFAR10.h5')

res = pd.DataFrame(model.history.history)
res[['loss', 'val_loss']].plot()
plt.show()
