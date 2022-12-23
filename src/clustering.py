import os
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn import cluster


n_components = 50
n_clusters = 20
BINARY = False

data = pd.read_csv("result/loc_pivot.csv", index_col='imsi').fillna(0)
poi_classes = list(data.columns)
data = data.to_numpy()

if BINARY:
    data[np.where(data>0)] = 1
else:
    data = data+1
    data = np.log10(data)

decomposer = TruncatedSVD(n_components = n_components, n_iter = 60)
# decomposer = NMF(n_components = n_components, init= 'random', max_iter = 400)

user_vector = decomposer.fit_transform(data)
poi_vector = decomposer.components_.T

# clusterer = cluster.KMeans(n_clusters=n_clusters)
clusterer = cluster.SpectralClustering(n_clusters=n_clusters)

poi_group = clusterer.fit_predict(poi_vector)

print(poi_group)

group_to_poi_classes_dict = {group:np.array(poi_classes)[np.where(poi_group==group)].tolist() for group in set(poi_group.tolist())}

for group in group_to_poi_classes_dict:
    print(f"group: {group}")
    for class_ in group_to_poi_classes_dict[group]:
        print(f"   class name: {class_}")