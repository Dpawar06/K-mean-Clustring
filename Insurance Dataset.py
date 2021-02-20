
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import  KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
#================================================================================================
# Analyze the information given in the following 'Insurance Policy dataset' to create clusters
# of persons falling in the same type
#==============================================================================================

Insurance = pd.read_csv("H:\DATA SCIENCE\Modules\Module 13 Unsupervise Data Mining K Maping\Insurance Dataset.csv\Insurance Dataset.csv")
# Summary
Insurance.columns
Insurance.describe
Insurance.shape
Insurance.isnull().sum()
Insurance.head(10)
 # Normalization function
def norm_func(i) :
    x = i - i.min() / i.max() - i.min()
    return(x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Insurance.iloc[ :, :])
df_norm

###### screw plot or elbow curve ############
k = list(range(2,8))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    WSS = [] # variable for storing within sum of squares for each cluster
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS)) # or TWSS.
TWSS
# Scree plot
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters
model=KMeans(n_clusters=5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row
md=pd.Series(model.labels_)  # converting numpy array into pandas series object
Insurance['clust']=md # creating a  new column and assigning it to new column
df_norm.head()


Insurance.iloc[:,1:7].groupby(Insurance.clust).mean()
Insurance.head()
