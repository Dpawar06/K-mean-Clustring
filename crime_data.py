
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Assignment 13 CrimeData
#=================================================================================================
# Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.
#==================================================================================================

Crime =pd.read_csv("H:\DATA SCIENCE\Modules\Module 13 Unsupervise Data Mining K Maping\crime_data.csv\crime_data.csv")
# Summary and Data Clining
Crime.shape
Crime.describe
Crime.isnull().sum()
Crime.head(10)
Crime.columns
Crime.info
Crimes = Crime.drop(Crime.columns[0], axis = 1)
# Normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()	-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Crimes.iloc[:,:]) # Normalizing data means converting data into 0 to 1 range or minimizing scale
df_norm
df_norm.head(10) # top 10 data


############ screw plot or elbow curve ######################################
from scipy.spatial.distance import cdist
k = list (range(2, 12))
k

TWSS = [] # TWSS  Variable  for storing total within sum of square  value of each cluster

for i in k :
    Kmeans = KMeans(n_clusters = i )
    Kmeans.fit(df_norm)
    TWSS.append(Kmeans.inertia_)
TWSS
#######################
plt.plot(k, TWSS, 'bx-')
plt.title("Crime Data elbow curve")
plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
plt.show() #Cluster size is =4


# Selecting 4 clusters from the above scree plot which is the optimum number of clusters
model=KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0) # create model object
model.fit(df_norm) # fit the model object to data
print(model.cluster_centers_) # print location of clusters learned by model objecet

pred_y = model.fit_predict(df_norm) # save the new cluster for chart
pred_y
model.labels_ # getting the labels of clusters assigned to each row
md=pd.Series(model.labels_)  # converting numpy array into pandas series object
Crimes['clust']= md # creating a  new column and assigning it to new column
Crimes.head()


# kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_norm
plt.scatter(df_norm["Murder"], df_norm["Assault"], df_norm["UrbanPop"],df_norm["Rape"],  model.cluster_centers_[:, 1], s =300, c='red')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='red')
plt.scatter(df_norm[pred_y ==0,0], df_norm[pred_y == 0,1], s=100, c='red')
plt.scatter(df_norm[pred_y ==1,0], df_norm[pred_y == 1,1], s=100, c='black')
plt.scatter(df_norm[pred_y ==2,0], df_norm[pred_y == 2,1], s=100, c='blue')
plt.scatter(df_norm[pred_y ==3,0], df_norm[pred_y == 3,1], s=100, c='cyan')
plt.show()


Crimes.iloc[:,1:7].groupby(Crimes.clust).mean()
Crimes.head()



