# Synthetic Fe-Cr APT Data Generator

The synthetic Atom Probe Tomography (APT) datasets generator for Fe-Cr alloys with Cr-rich clusters are prepared in both MATLAB scripts and Python files. It is designed for:

    1. Machine Learning (ML) training (e.g., clustering, segmentation,  anomaly detection)

    2. Algorithm benchmarking when real APT data is limited

    3. Parameter studies for understanding clustering behavior


## 1. Why synthetic data?
Experimental APT data collection is time-consuming, expensive, and often limited by:

- Noise and reconstruction artifacts

- Small dataset size

- Limited labeling (ground truth unavailable)

Synthetic datasets allow:

✔ Controlled composition and structure

✔ Known ground truth labels for validation

✔ Flexible size and cluster distribution

## 2. Code Overview

The main steps:

    1. Cluster size generation (via generateClusterSizes)

    2. Cluster center placement within the simulation box (generateClsuterCenters)

    3. Cr cluster atom generation (generateCrClusters)

    4. Background atom generation for Fe and Cr (generateBackground)

    5. Combine and save the dataset as a .mat file and .txt file

Each dataset contains:

- [X, Y, Z] coordinates in nanometers

- Labels (starting from 1)

## 3. User Input Parameters

These parameters must be set by the user at the top of the script:


| **Parameter**           | **Description**                             |**Example**                                 |
|-------------------------|---------------------------------------------|---------------------------------------------|
| Dim                     | Simulation volume [X, Y, Z] in nm           | [80, 80, 80]                                |
| TotalNumPoints          | Total number of atoms (Fe + Cr)             | 7,000,000                                   |
| CrPercentage            | Fraction of Cr atoms                        | 0.12 (12%)                                  |
| Density                 | Cluster density (clusters/m³)               | 9.5e23                                      |
| NumClusters             | Number of Cr-rich clusters                  | 100                                         |
| ClusterSizeInfo         | NumClusters, Cluster radius (nm) and fluctuation (nm) in X direction, Cluster radius (nm) and fluctuation (nm) in Y direction, Cluster radius (nm) and fluctuation (nm) in Z direction                | [NumClusters, 1.5, 0.2, 1.5, 0.2, 1.5, 0.1]  |
| MinClusterSeparation    | Minimum distance between clusters (nm)      | 5                                           |
| OutputFolder            | Folder to save datasets                     | './SyntheticData'                           |


## 4. How to Choose Reasonable Values

When setting parameters for your simulation, consider the following guidelines:

- **Dim**: Choose dimensions that are large enough to capture relevant phenomena but small enough to keep computation manageable.Choose based on memory constraints. Millions of atoms are typical for APT.
- **TotalNumPoints**: Ensure the number of atoms reflects realistic densities and system sizes.
- **CrPercentage**: Base this on experimental data or desired alloy composition.
- **Density**: Use known atomic densities for the material system.
- **NumClusters**: Select a number that allows for statistical analysis without overcrowding the simulation volume.
- **ClusterSizeInfo**: Adjust size and fluctuation to reflect realistic microstructural features.

    First value = number of clusters  
    Next pairs = mean radius ± fluctuation for X, Y, Z (in nm)
- **MinClusterSeparation**: Ensure clusters are sufficiently spaced to avoid overlap  (e.g., 3–5× cluster radius).
- **OutputFolder**: Choose a directory with sufficient storage and clear naming conventions.

Adjust these values iteratively based on simulation results and physical intuition.

## 5. Customization Tips

- Adjust cluster shape randomness via ClusterSizeInfo.

- Update RemainingFe formula if Fe distribution should vary.

- Enable visualization by uncommenting the scatter3 section.

## 6. Dependencies

- Python: Step1_generateclusters.py  
- MATLAB: Step1_generateclusters.m (main script)  
    Functions:  
    - generateClusterSizes.m  
    - generateClusterCenters.m  
    - generateCrClusters.m  
    - generateBackground.m


