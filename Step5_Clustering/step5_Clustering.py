# ============================================
#  K-Means + HDBSCAN Clustering Script (Python)
#  Description:
#    - Loads raw 3D coordinates and model predictions 
#    - Preliminary clustering with K-Means
#    - HDBSCAN refinement
# ============================================

import os
import sys
import math
import json
import numpy as np
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import h5py
from scipy.io import loadmat
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# ================================
# 1) USER INPUTS
# ================================
rawDataFile      = "subvolume.txt"               # <-- Raw 3D coordinates .mat; expects a variable containing Nx3 coords
predictionFile   = "predictions.mat"   # <-- .mat with model predictions (vector or scalar)
threshold        = 2.0                      # <-- distance threshold for assigning clusters
minPersistence   = 0.06                     # <-- persistence cutoff for HDBSCAN filtering


# Choose HDBSCAN mode:
HDBSCAN_MODE     = "native"                 # "native" (uses hdbscan lib) or "external" (call your script)


# External-script settings (used only if HDBSCAN_MODE == "external")
# pythonEnvPath    = "/home/tangy/miniforge3/envs/APT/bin/python"  # path to python exe
# hdbscanScript    = "hdbscanImanSecond_9.py"                      # your external script
tempDataFile     = "TempHDBSCANfile.txt"
tempParamFile    = "TempHDBSCANparameters.txt"


def _counts_from_positive_labels(labels: np.ndarray) -> np.ndarray:
    #"""
    #Equivalent to MATLAB accumarray(clusterLabels(clusterLabels>0), 1)
    #"""
    pos = labels[labels > 0]
    if pos.size == 0:
        return np.array([], dtype=int)
    m = int(pos.max())
    counts = np.bincount(pos, minlength=m + 1)  # index 0 unused
    return counts[1:]  # drop 0


# ================================
# 2) LOAD DATA
# ================================
print("Loading data...")
#raw_mat = loadmat(rawDataFile)
#pred_mat = loadmat(predictionFile)

gt=np.loadtxt(rawDataFile)
pred_mat = loadmat(predictionFile)
predictions = pred_mat['predictions']
pred_n = np.round(predictions[0][0])



# KMeans over coordinates
kmeans = KMeans(n_clusters=int(pred_n), n_init="auto", random_state=0)
kmeans.fit(gt[:, :3])
C = kmeans.cluster_centers_


# Compute distances to centroids, assign closest, mark outliers by threshold
D = cdist(gt[:, :3], C, metric="euclidean")
minDist = D.min(axis=1)
idx = D.argmin(axis=1)


clusterLabels = idx.astype(int) + 1  # make them 1..K like MATLAB
clusterLabels[minDist > threshold] = -1


counts = _counts_from_positive_labels(clusterLabels)
label = clusterLabels.copy()
print(f"Preliminary clustering complete. Found {len(counts)} clusters.")

# ================================
# 4) DETERMINE HDBSCAN PARAMETERS
# ================================
if counts.size == 0:
    # no clusters survived the threshold; nothing to refine
    print("No valid clusters from K-Means. Exiting early.")
    hdbscanCluster_CNN: List[Cluster] = []
else:
    m = int(np.min(counts))            # min_cluster_size
    n = int(np.round(0.1 * m))         # min_samples
    print(f"HDBSCAN parameters: min_cluster_size = {m}, min_samples = {n}")

# ================================
# 5) HDBSCAN 
# ================================
import sys
import hdbscan
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Save the first 3 columns of gt to a text file with space delimiter
np.savetxt(tempDataFile, gt[:, :3], delimiter=' ', fmt='%.6f')

# Write m and n to a parameter file
with open(tempParamFile, 'w') as f:
    f.write(f"{m}\n")
    f.write(f"{n}")

    
X=np.genfromtxt(tempDataFile)
Y=np.genfromtxt(tempParamFile)
MinClusterSize=int(Y[0])
MinSamples=int(Y[1])

clusterer = hdbscan.HDBSCAN(min_cluster_size=MinClusterSize,min_samples=MinSamples,
                            cluster_selection_method='eom',approx_min_span_tree=False,core_dist_n_jobs=1)
clusterer.fit(X)
tree=clusterer.condensed_tree_.plot(select_clusters=True,selection_palette=sns.color_palette())
np.savetxt('Labels.txt',clusterer.labels_, fmt='%d',comments='')
np.savetxt('Persistence.txt',clusterer.cluster_persistence_,comments='')
np.savetxt('Probabilities.txt',clusterer.probabilities_,comments='')  


# ================================
# 6) PROCESS HDBSCAN RESULTS (native)
# ================================

# Load data from text files
hdbscan_labels = np.loadtxt('Labels.txt')
hdbscan_persistence = np.loadtxt('Persistence.txt')
hdbscan_probabilities = np.loadtxt('Probabilities.txt')
PositionDataset = np.loadtxt(tempDataFile)
SiZePositionDatasetCol=PositionDataset.shape[1] 

import numpy as np

NoiseLabel = -1
UniqueLabels = np.unique(hdbscan_labels)

hdbscanCluster = []

# Determine if noise label is present
has_noise = NoiseLabel in UniqueLabels

# Adjust loop range depending on presence of noise
loop_range = range(len(UniqueLabels) - 1) if has_noise else range(len(UniqueLabels))

for i in loop_range:
    label = i
    # Find indices where hdbscan_labels == i
    row_indices = np.where(hdbscan_labels == label)[0]

    cluster_info = {
        'labels': label,
        'probabilities': hdbscan_probabilities[row_indices],
        'persistence': hdbscan_persistence[i],
        'atomPositions': PositionDataset[row_indices, :SiZePositionDatasetCol],
        'clustersize': len(row_indices)
    }

    hdbscanCluster.append(cluster_info)


