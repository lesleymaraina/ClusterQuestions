######################################################
#IMPORT
######################################################
import pandas as pd
from sklearn.cluster import KMeans
import csv
import numpy as np
import numpy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.spatial
from sklearn.cluster import DBSCAN

######################################################
#DATA UPLOAD 
######################################################
reader=csv.reader(open("/Users/lmc2/Downloads/114_congress2.csv","rb"),delimiter=',')
X=list(reader)
result=numpy.array(X).astype('float')
votes = pd.read_csv("/Users/lmc2/Downloads/114_congress.csv", error_bad_lines=False)


######################################################
#DATA ANALYSIS
######################################################
#KMeans Model
kmeans_model = KMeans(n_clusters=2, random_state=1).fit(votes.iloc[:, 3:])
labeled = kmeans_model.labels_
votes['label'] = labeled

#Find Centroids
centroids = kmeans_model.cluster_centers_
sqdists = scipy.spatial.distance.cdist(centroids, result, 'sqeuclidean')


#List the distance of each data point to each cluster
#List corresponding index
#TODO: How to list the "name" that corresponds to each distance?
clusters = np.argmin(sqdists, axis=0)
df2 = pd.DataFrame(sqdists)
print df2.head()


#DBSCAN
#TODO: List data points in dense cluster regions
dbsc = DBSCAN(eps = .5, min_samples = 15).fit(votes)


