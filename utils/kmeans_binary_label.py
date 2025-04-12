# kmeans_binary_label

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Assume ic50_df is a DataFrame with rows as cell lines, columns as drugs
# Each value is IC50 (float)

binary_labels = pd.DataFrame(index=ic50_df.index)
for drug in ic50_df.columns:
    ic50_values = ic50_df[drug].dropna().values.reshape(-1, 1)
    if len(ic50_values) < 2:
        continue  # skip if not enough data
    
    # Perform K-means clustering with 2 clusters (responder vs non-responder)
    kmeans = KMeans(n_clusters=2, random_state=0, init='k-means++', n_init='auto', max_iter=300, tol=0.0001).fit(ic50_values)
    clusters = kmeans.labels_
    # 取出小的index
    responder_cluster = np.argmin(kmeans.cluster_centers_)
    # 將 1 = responder, 0 = non-responder 
    binary = (clusters == responder_cluster).astype(int)


    
    # Assign labels back to DataFrame
    binary_labels[drug] = pd.Series(binary, index=ic50_df[drug].dropna().index)
    print("binary_labels[drug],",binary_labels[drug])