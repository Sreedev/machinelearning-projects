#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:58:58 2019

@author: admin
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import numpy as np

#Reading crime data csv
mall_customers = pd.read_csv("Mall_Customers.csv")

#Normalizing all the values and create a new dataset since it has a lot of variations
def norm_func(i):
    x = (i-i.mean())/(i.std())
    return (x)
mall_customers_norm = norm_func(mall_customers.iloc[:,2:])

#Creating data for plotting elbow curve and plotting
k = list(range(1,20))
k
TWSS = [] 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(mall_customers_norm)
    WSS = []
    for j in range(i):
        WSS.append(sum(cdist(mall_customers_norm.iloc[kmeans.labels_==j,:],
                             kmeans.cluster_centers_[j].reshape(1,mall_customers_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

#Creating clurring model with the optimal number of clusters which we find out from elbow curve.
model=KMeans(n_clusters=7) 
model.fit(mall_customers_norm)

#Numbering each row with the specific cluster
model.labels_ 
md=pd.Series(model.labels_)  

#Adding the cluster number to the main dataset and re arraging the coloumns
mall_customers['clust']=md  
mall_customers = mall_customers.iloc[:,[5,0,1,2,3,4]]

#Finding the mean of all the values according to each cluster
mall_customers.iloc[:,[0,3,4,5]].groupby(mall_customers.clust).mean()
plt.scatter(mall_customers.iloc[:,0], mall_customers.iloc[:,3], 
            s = 4, c = 'red', label = 'Age')
plt.scatter(mall_customers.iloc[:,0], mall_customers.iloc[:,4], 
            s = 8, c = 'blue', label = 'Income')
plt.scatter(mall_customers.iloc[:,0], mall_customers.iloc[:,5], 
            s = 12, c = 'black', label = 'Spend score')
plt.title('Mall customers cluster with Age')
plt.xlabel('Cluster')
plt.ylabel('Age, Income and spendscore')





