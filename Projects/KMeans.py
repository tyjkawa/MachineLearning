import pandas as pd
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from sklearn.cluster  import KMeans


#Loading the dataset into a dataframe.
csv_file = "wineQualityReds.csv"
wineData = pd.read_csv(csv_file, header=0)
print(wineData)


# Cleaning the Data
wineData.drop(wineData.columns[0], axis = 1, inplace=True)
print(wineData)
quality_column = wineData['quality']
wineData.drop('quality', axis = 1, inplace=True)
print(quality_column)
norm = Normalizer()
wineData_norm = pd.DataFrame(norm.transform(wineData), columns = wineData.columns)


# Creating a range of k values from 1:11 for KMeans clustering. Iterating on the k values and store the inertia for each clustering in a list.

ks = range(1,12)
inertia = []

for k in ks:
  model = KMeans(n_clusters=k, random_state = 2023, n_init='auto')
  model.fit(wineData_norm)
  inertia.append(model.inertia_)
print(inertia)

# Plotting the chart of inertia vs number of clusters

plt.plot(ks, inertia, '-o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Inertia vs Number of Clusters')

plt.show()


# Picked the optimal # of K clusters, predicted, illustarted the findings

model = KMeans(n_clusters=5, random_state = 2023)
model.fit(wineData_norm)
wineLabels = model.predict(wineData_norm)
print(wineLabels)

wineData_norm['Cluster Label'] = wineLabels
print(wineData_norm)

wineData_norm['quality'] = quality_column
print(wineData_norm)

crosstab = pd.crosstab(wineData_norm['Cluster Label'], wineData_norm['quality'])
print(crosstab)



