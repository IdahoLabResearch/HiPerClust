# K-Means + HDBSCAN Clustering for 3D Point Data

This module contains the script for clustering 3D point cloud data using a two-step process:

Preliminary Clustering with K-Means

Refined Clustering with HDBSCAN (run via Python)

The pipeline is designed for datasets where an initial estimate of the number of clusters is available from the machine learning model (e.g., ConvNeXt-Tiny, ResNet-50).

## 1. Code Overview

This folder contains the following files:

- step5_clustering.py

    Python script for clustering

- step5_clustering.m and hdnscanImanScond.py

    MATLAB script for clustering

## 2. Main Steps in the Script

 1. Load Data

    Loads raw 3D point coordinates (data) and prediction file (predictions).

 2. Preliminary Clustering with K-Means

    Uses predicted number of clusters (pred_n) for K-Means.

    Computes distances and removes outliers beyond threshold.

 3. Determine HDBSCAN Parameters

    min_cluster_size = min(K-Means cluster size)

    min_samples = 10% of min_cluster_size

 4. Prepare Files for Python

    Writes input points and parameters for the HDBSCAN script.

 5. Run HDBSCAN (Python)

    Calls Python environment to execute hdbscanImanSecond.py.

 6. Post-process HDBSCAN Output

    Reads results using hdbscanPostProcess.
    
    Filters clusters by persistence and size.

## 3. How to Use

1. Prepare Files

    Place your raw 3D data.

    Place your model prediction file (e.g., ConvNeXtLarge_12.mat or ResNet50_12.mat).

    Ensure Python environment and script hdbscanImanSecond.py are available.

2. Set User Inputs in the script

    rawDataFile      = 'subbolume.txt';         % Raw 3D coordinates file
    predictionFile   = 'predictions.mat';       % Predictions file
    pythonEnvPath    = '/path/to/python';       % Path to Python executable
    hdbscanScript    = 'hdbscanImanSecond.py';% HDBSCAN Python script


## 4. Example Output

The function processHDBSCANResults(dataFile) returns a structure array named clusters. Each element in this array represents a single detected cluster (excluding noise).
Each clusters contains the following fields:

| **Field Name**   | **Description**                                                                 |
|------------------|---------------------------------------------------------------------------------|
| labels         | Cluster ID (integer, starting from 0).                                          |
| probabilities  | Membership probability for each point in this cluster (vector).                 |
| persistence    | Persistence score of the cluster (higher = more stable cluster).                |
| atomPositions  | Nx3 matrix of coordinates for all points in this cluster.                       |
| clustersize    | Number of points in this cluster (integer).                                     |

