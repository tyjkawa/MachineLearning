import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


#Reading and cleaning data

data = pd.read_csv('A_Z Handwritten Data.csv')

print(data)
print(data.label.unique())
# The Target is the label column. The label column contains values from [0-25] which each represent a letter in the alphabet. This is what we are trying to learn and predict.


X = data.iloc[:,1:]
print(X)

y = data["label"]
print(y)


print(X.shape)
print(y.shape)


#The target variable values are numbers which represent letters such that A=0, B=1, etc.
word_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}

y = y.map(word_dict)

sb.countplot(x=y)

random_64_letters = data.sample(n=64)

for i in range(64):
  placeholder = random_64_letters.iloc[i,1:]
  placeholder = np.array(placeholder)
  placeholder = placeholder.reshape(28,28)
  plt.subplot(8, 8, i+1)
  plt.imshow(placeholder, cmap='gray')
  plt.axis('off')
  plt.text(10,-4, word_dict[random_64_letters.iloc[i,0]])

plt.tight_layout()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .3, random_state = 2023, stratify = y)

X_train = X_train/255
X_test = X_test/255

model = MLPClassifier(hidden_layer_sizes=(100,100,100), activation='relu',
                 max_iter = 50, alpha = 0.001, solver='adam',
                      random_state = 2023, learning_rate_init=0.01, verbose=True)

model.fit(X_train, y_train)

plt.plot(model.loss_curve_)

y_pred = model.predict(X_test)
print('The model accuracy on the test data is: ', model.score(X_test, y_test))

cm = confusion_matrix(y_pred, y_test)
plt.figure(figsize=(20,20))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=word_dict.values()).plot()
plt.show()

sample = X_test.iloc[[0]]
prediction = model.predict(sample)

picture = X_test.iloc[[0]]
picture = np.array(picture)
picture = picture.reshape(28, 28)
plt.imshow(picture, cmap='gray')

print("The predicted letter is", prediction[0] , "and the actual letter is", y_test.iloc[0])

sample = X_test.loc[[83742]]
prediction = model.predict(sample)

sample = np.array(sample)
picture = sample.reshape(28, 28)
plt.imshow(picture, cmap='gray')

print("The failed predicted letter is", prediction[0] , "and the actual letter is", y_test.loc[83742])
