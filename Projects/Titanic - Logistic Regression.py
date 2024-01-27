import scikitplot as skplt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

#Extracting and cleaning data

titanicData = pd.read_csv("Titanic.csv")
print(titanicData)

titanicData = titanicData.drop(["Passenger"], axis=1) #axis = 1 means delete the column
print(titanicData)

titanicData = pd.get_dummies(titanicData, columns=["Class", "Sex", "Age"], drop_first=True)
print(titanicData)

y = titanicData['Survived']
X = titanicData.drop(columns=['Survived'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=2023, stratify=y)


# Creating model and fitting the training data to a logistic regression model

LogReg = LogisticRegression(max_iter=500)
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)
print('The accuracy of the model is: ', metrics.accuracy_score(y_test, y_pred))

# Plotting the lift curve and confusion matrix

y_probas = LogReg.predict_proba(X_test)
skplt.metrics.plot_lift_curve(y_test, y_probas)
plt.show()

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LogReg.classes_).plot()
plt.show()

# Testing data with a prediction value of the survivability of a male adult passenger traveling in 3rd class.

data = { 'Class_2nd': [0],
         'Class_3rd': [1],
         'Class_Crew': [0],
         'Sex_Male':[1],
         'Age_Child': [0]}

individual = pd.DataFrame(data)

individual_predict = LogReg.predict(individual)

print(individual_predict)