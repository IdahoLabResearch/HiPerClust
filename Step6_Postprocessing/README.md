# Post-Processing of Detected Clusters

After HDBSCAN clustering, material scientists often require quantitative characteristics of the detected clusters, such as:

- Cluster Radius
- Number Density
- Volume Fraction

These metrics provide insights into the size, spatial distribution, and overall contribution of precipitates or clusters in the material microstructure.

The radius \( R^{\text{Cr}} \) of the Cr-rich clusters is calculated using the spherical equivalent radius [Kolli et al., 2007], given by:

$$
R^{\text{Cr}} = \sqrt[3]{\frac{3N_{\text{prec}}}{4\pi \rho Q}},
$$

where \( N_{\text{prec}} \) is the number of detected Cr atoms in each precipitate, \( \rho \) is the density of the precipitate (assumed to be identical to \( \alpha \)-Fe), \( 84.3\, \text{atoms nm}^{-3} \), and \( Q \) is the detection efficiency of the atom probe in use (taken to be 0.36).

The number density \( N_d^{\text{Cr}} \) is determined by:

$$
N_d^{\text{Cr}} = \frac{N_p \rho Q}{N_{\text{tot}}},
$$

where \( N_p \) is the number of measured precipitates in the reconstructed volume, and \( N_{\text{tot}} \) is the total number of atoms in the reconstructed volume.

The volume fraction [\( \varphi^{\text{Cr}} \)](README.md) is defined by:

$$
\varphi^{\text{Cr}} = \frac{\sum N_{\text{prec}}}{N_{\text{tot}}},
$$

where \( \sum N_{\text{prec}} \) is the sum of the atoms contained in the Cr precipitates within the analyzed volume, and \( N_{\text{tot}} \) is the total number of atoms in the reconstructed volume.


## MATLAB Example Code

The following MATLAB script demonstrates how to compute number density, cluster radius, and volume fraction using the clusters structure returned by processHDBSCANResults.

```matlab
%% ================================
% Post-Processing of Cluster Properties
% =================================

% INPUTS:
% clusters      - structure array from processHDBSCANResults
% subvolumeSize - volume of the analyzed region (in nm^3)
% dataPoints    - all points in the analyzed region (Nx3 matrix)

%% Parameters
Q = 0.36;                % Detection efficiency
rho = 84.3;              % Atomic density of alpha-Fe (atoms/nm^3)
subvolumeSize=20^3;      % subvolumesize in nm^3

%% Number Density (clusters per m^3)
numClusters = numel(clusters);
numberDensity = numClusters / subvolumeSize *1e27;   % clusters per m^3

%% Cluster Radius for Each Cluster (nm)
clusterRadii = zeros(numClusters,1);
for i = 1:numClusters
    numAtoms = clusters(i).clustersize; % Number of atoms in the cluster
    clusterRadii(i) = ((3 * numAtoms) / (4 * pi * rho * Q))^(1/3);
end

%% Volume Fraction of Clusters
totalClusterAtoms = sum([clusters.clustersize]); % Sum of all cluster atoms
totalAtoms = size(dataPoints, 1);               % Total atoms in the , in the filtered data
CrPercent = 0.09 % The Cr percentage on the raw 3D data, e.g. 9%
volumeFraction = totalClusterAtoms / (totalAtoms ./ CrPercent) ;

%% Display Results
fprintf('Number Density: %.4e clusters/nm^3\n', numberDensity);
fprintf('Average Cluster Radius: %.3f nm\n', mean(clusterRadii));
fprintf('Volume Fraction: %.4f\n', volumeFraction);
```

## Python Example Code

The following python script demonstrates how to compute number density, cluster radius, and volume fraction using the clusters structure returned by processHDBSCANResults.

```python
import numpy as np

# ================================
# Post-Processing of Cluster Properties
# ================================

# INPUTS:
# hdbscanCluster - list of dictionaries from previous step
# subvolume_size - volume of the analyzed region (in nm^3)
# PositionDataset - all points in the analyzed region (Nx3 matrix)

# ---- Parameters ----
Q = 0.36 # Detection efficiency
rho = 84.3 # Atomic density of alpha-Fe (atoms/nm^3)
CrPercent = 0.09 # Cr percentage in the raw 3D data (e.g., 9%)
subvolume_size=40^3 # subvolume size in nm^3

# ---- Number Density (clusters per m^3) ----
num_clusters = len(hdbscanCluster)
number_density = num_clusters / subvolume_size *1e27 # clusters per m^3

# ---- Cluster Radius for Each Cluster (nm) ----
cluster_radii = np.zeros(num_clusters)
for i, cluster in enumerate(hdbscanCluster):
 num_atoms = cluster['clustersize'] # Number of atoms in the cluster
 cluster_radii[i] = ((3 * num_atoms) / (4 * np.pi * rho * Q)) ** (1/3)

# ---- Volume Fraction of Clusters ----
total_cluster_atoms = sum(cluster['clustersize'] for cluster in hdbscanCluster)
total_atoms = PositionDataset.shape[0]
volume_fraction = total_cluster_atoms / (total_atoms / CrPercent)

# ---- Display Results ----
print(f"Number Density: {number_density:.4e} clusters/m^3")
print(f"Average Cluster Radius: {np.mean(cluster_radii):.3f} nm")
print(f"Volume Fraction: {volume_fraction:.4f}")
```

A simple python code to visualize the clustersusing different colors.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example: replace this with your actual data
# data = np.loadtxt('your_data.txt')  # or however you load your nx4 matrix
# Assuming data is an (n x 4) numpy array
# Columns: x, y, z, cluster_label
# Example dummy data:
data = np.loadtxt('example_plot.txt')

def plot_clusters(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    labels = data[:, 3].astype(int)

    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        ax.scatter(x[mask], y[mask], z[mask], label=f'Cluster {cluster_id}', s=20)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('3D Cluster Visualization')
    plt.show()

# Call the function with your data
plot_clusters(data)

```

<p align="center">
    <img src="Images/step6_output.png" alt="Image" Width="25%">
    <br>
    <em> Figure 1: Color-coded visualization of APT clusters. </em>
</p>
