
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import  KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

#============================================================================================
# K mean Clustering on Airline Data set
#==============================================================================================
import xlrd # loading the data set in excel file
Airline = pd.read_excel("H://DATA SCIENCE//Modules//Module 13 Unsupervise Data Mining K Maping//EastWestAirlines Dataset 2.xlsx//EastWestAirlines.xlsx", sheet_name = "data")
print(Airline)


# Data mining
Airline1 =  Airline.drop(["ID#"], axis=1 )
Airline1.info

Airline1.isnull().sum()
Airline1['Balance'].unique()
#

# summary
Airline1.describe
Airline1.shape


# Normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()	-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Airline1.iloc[:,:]) # Normalizing data means converting data into 0 to 1 range or minimizing scale
df_norm
df_norm.head(10) # top 10 data

###### screw plot or elbow curve ############
from scipy.spatial.distance import cdist


k = list(range(2,12)) # this is the elbow range
k
TWSS = []# Variable for storing totalwithin sum of square for each cluster
for i in k:
    kmeans =KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)

TWSS
###############################################################
#or  Scarttee plot
#
plt.plot(k, TWSS, 'bx-')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distances')
plt.title('elbow method for optimal k')
plt.show()

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters
model=KMeans(n_clusters=5)
model.fit(df_norm)
model
model.cluster_centers_
model.labels_ # getting the labels of clusters assigned to each row
md = pd.Series(model.labels_)  # converting numpy array into pandas series object
Airline1['clust']= md # creating a  new column and assigning it to new column
Airline1.head()
df_norm.head()

Airline1.iloc[ :, 2:12 ].groupby(Airline1.clust).mean()

# or same as line no. 63 with full algorithem
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans

Airline1.iloc[:,1:7].groupby(Airline1.clust).mean()
Airline1.head()

























