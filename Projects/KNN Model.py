import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#Extracting and cleaning data

diabetesData = pd.read_csv("diabetes.csv")
print(diabetesData)

X = diabetesData.iloc[:, 0:8]
y = diabetesData.iloc[:, 8]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_trainA, X_40, y_trainA, y_40 = train_test_split(X, y, test_size=0.4, random_state=2023, stratify=y)
X_trainB, X_test, y_trainB, y_test = train_test_split(X_40, y_40, test_size=0.5, random_state=2023, stratify=y_40)


# Creating Model and Training

neighbors = np.arange(1, 31)
trainA_accuracy = np.empty(30)
trainB_accuracy = np.empty(30)

for k in neighbors:
  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(X_trainA, y_trainA)
  trainA_accuracy[k-1] =model.score(X_trainA, y_trainA)
  trainB_accuracy[k-1] = model.score(X_trainB, y_trainB)

  #A -- Train
  #B -- Validation (Trying to understand if the model is being trained properly)
  #C -- Test


# Displaying the accuracies

print("Accuracy for Training A:", trainA_accuracy)
print("Accuracy for Training B:", trainB_accuracy)

plt.plot(neighbors, trainA_accuracy, label = 'Training A accuracy')
plt.plot(neighbors, trainB_accuracy, label = 'Training B accuracy')
plt.title("Accuracys")
plt.legend()
plt.xlabel('# of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# Picking the best K and displaying model findings

model = KNeighborsClassifier(n_neighbors=15)
model.fit(X_trainA, y_trainA)
y_pred = model.predict(X_test)
test_accuracy = model.score(X_test, y_test)

print("The Test Accuracy with 15 Ks:", metrics.accuracy_score(y_test, y_pred))

print(y_test)
print(y_pred)
