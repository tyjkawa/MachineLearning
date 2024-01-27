import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sb


keras.utils.set_random_seed(2023)
tf.config.experimental.enable_op_determinism()

#Reading and cleaning data
fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

X_test = test_images
X_train = train_images

y_test = test_labels
y_train = train_labels

print("Feature Test Shape:", X_test.shape)
print("Feature Train Shape:", X_train.shape)
print("Target Test Shape:", y_test.shape)
print("Target Train Shape:",y_train.shape)

#The target variable values are numbers which represent different clothing items
# 0 = T-shirt/top
# 1 = Trouser
# 2 = Pullover
# 3 = Dress
# 4 = Coat
# 5 = Sandal
# 6 = Shirt
# 7 = Sneaker
# 8 = Bag
# 9 = Ankle boot

label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

y_test_mapped = [label_dict[label] for label in y_test]
y_train_mapped = [label_dict[label] for label in y_train]

#Display data

sb.countplot(x=y_train_mapped)
plt.xticks(rotation=45, ha='right')
plt.show()

random_sample = np.random.choice(X_train.shape[0], size = 25, replace = False)

random_sample_features = X_train[random_sample]
random_sample_labels = y_train[random_sample]
random_sample_labels_mapped = [label_dict[label] for label in random_sample_labels]

for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.imshow(random_sample_features[i], cmap="gray")
  plt.axis('off')
  # plt.text(3,-3, random_sample_labels_mapped[i])
  plt.text(15, 40, random_sample_labels_mapped[i], ha='center')
  # ha='center' ensures that the text is horizontally centered at the specified x-coordinate.

plt.tight_layout()

#Cleaning Data

X_train = X_train / 255
X_test = X_test / 255

#Creating the Model
model = keras.models.Sequential()

#Creating the Flatten Layer
model.add(keras.layers.Flatten(input_shape=(28,28)))

#Creating Two Dense Layers
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))

# Adding a Dense Layer as Output Layer
model.add(keras.layers.Dense(10, activation = "softmax"))

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer ='sgd', metrics = ['accuracy'])

# Fitting the model

callback = keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.001, patience = 5)

history = model.fit(X_train, y_train, epochs = 100, verbose = True, callbacks = [callback])

# Display the results

pd.DataFrame(history.history).plot()

accuracy = history.history['accuracy'][-1]
print(accuracy)

y_actual_row_one = y_test[0]
y_actual_label = label_dict[y_actual_row_one]

row_one = X_test[0]
row_one_reshaped = row_one.reshape((1, 28, 28))
y_pred_row_one = model.predict(row_one_reshaped)
y_pred_row_one = np.argmax(y_pred_row_one, axis = 1)
y_pred_row_one_label = label_dict[y_pred_row_one[0]]

plt.imshow(row_one,cmap="gray")

plt.text(15, -3, "The Actual Item is " + y_actual_label, ha='center', color='black', fontsize=10)
plt.text(15, 32, "The Predicted Item is " + y_pred_row_one_label, ha='center', color='black', fontsize=10)

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1)
print(y_pred)

failed_indices = np.where(y_test != y_pred)[0]
print(failed_indices[0])

y_wrong = y_test[12]
y_wrong_label = label_dict[y_wrong]

row_wrong = X_test[12]
row_wrong_reshaped = row_wrong.reshape((1, 28, 28))
y_pred_wrong = model.predict(row_wrong_reshaped)
y_pred_wrong = np.argmax(y_pred_wrong, axis = 1)
y_pred_wrong_label = label_dict[y_pred_wrong[0]]

plt.imshow(row_wrong,cmap="gray")

plt.text(15, -3, "The failed predicted apparel is " + y_pred_wrong_label + " whereas the actual apparel is " + y_wrong_label, ha='center', color='black', fontsize=10)

