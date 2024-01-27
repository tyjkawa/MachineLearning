import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D


# Loading and Visualizing Data

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode="fine")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
               'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
               'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud',
               'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
               'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp',
               'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
               'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit',
               'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
               'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
               'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
               'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

plt.figure(figsize = [10,10])

for i in range(30):
  plt.subplot(5,6,i+1)
  plt.imshow(X_train[i])
  plt.xticks([])
  plt.yticks([])
  plt.xlabel(class_names[y_train[i, 0]])

# Scaling Data
X_train = X_train/255
X_test = X_test/255


# Creating the Model
model = Sequential()

# Feature Learning Part of Model
model.add(Conv2D(filters=100, kernel_size = (3,3), strides = (1,1), padding = 'same',
                 activation = 'relu', input_shape =(32, 32, 3)))

model.add(Conv2D(filters=100, kernel_size = (3,3), strides = (1,1), padding = 'same',
                 activation = 'relu'))

model.add(MaxPool2D(2,2))

model.add(Dropout(.2))

model.add(Conv2D(filters=100, kernel_size = (3,3), strides = (1,1), padding = 'same',
                 activation = 'relu'))

model.add(Conv2D(filters=50, kernel_size = (3,3), strides = (1,1), padding = 'same',
                 activation = 'relu'))

model.add(MaxPool2D(2,2))

model.add(Dropout(.25))

# Classification Part of Model
model.add(Flatten())

model.add(Dense(units = 750, activation = 'relu'))

model.add(Dropout(.3))

model.add(Dense(units = 500, activation = 'relu'))

model.add(Dropout(.3))

model.add(Dense(units = 100, activation = 'softmax'))

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')

# Training the Model
h = model.fit(X_train, y_train, epochs = 30, validation_data = (X_test, y_test))

#Displaying Results

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curves')

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curves')

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)

print(y_pred.shape)
print(y_test.shape)

plt.figure(figsize = [15,15])

for i in range(30):
  plt.subplot(5,6,i+1)
  plt.imshow(X_test[i])
  plt.xticks([])
  plt.yticks([])
  plt.title('Actual: %s \nPredicted: %s' % (class_names[y_test[i,0]], class_names[y_pred[i]]))

plt.subplots_adjust(hspace = 1)

placeholder = pd.DataFrame(y_test)

misclassified = placeholder[placeholder[0]!=y_pred]

random_sample = misclassified.sample(30)

random_sample_index = list(random_sample.index)

plt.figure(figsize = [15,15])

for i in range(30):
  plt.subplot(5,6,i+1)
  plt.imshow(X_test[random_sample_index[i]])
  plt.xticks([])
  plt.yticks([])
  plt.title('Actual: %s \nPredicted: %s' % (class_names[y_test[random_sample_index[i],0]],
                                            class_names[y_pred[random_sample_index[i]]]))

plt.subplots_adjust(hspace = 1)