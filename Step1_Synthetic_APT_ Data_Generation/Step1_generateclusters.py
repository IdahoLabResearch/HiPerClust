from __future__ import annotations
import os
import math
import warnings
import numpy as np
from typing import Tuple, List
from scipy.io import savemat
from scipy.spatial.distance import pdist

try:
    from tqdm import trange
except ImportError:
    def trange(*args, **kwargs):
        return range(*args, **kwargs)


# ===================== USER INPUT SECTION =====================
Dim = np.array([80.0, 80.0, 80.0]) # Volume dimensions [X, Y, Z] in nm
TotalNumPoints = int(7000000) # Total number of atoms (Fe + Cr)
CrPercentage = 0.12 # Chromium atomic fraction
    
Density = 9.5e23 # Atom density in atoms/m^3 (unused)
DensityError = 0.2e23 # Density uncertainty (unused)
    
NumClusters = 100
# clusterInfo: [NumClusters, rx_mean, rx_sigma, ry_mean, ry_sigma, rz_mean, rz_sigma]
clusterInfo = np.array([NumClusters, 1.5, 0.2, 1.5, 0.2, 1.5, 0.1], dtype=float)
    
MinClusterSeparation = 5.0 # Minimum separation between cluster centers (nm)
OutputFolder = './SyntheticData'
os.makedirs(OutputFolder, exist_ok=True)
    
# Derived values
TotalNumCr = int(round(CrPercentage * TotalNumPoints))
TotalNumFe = TotalNumPoints - TotalNumCr
    
# Extra knobs
RANDOM_SEED = 42
F_CLUSTERED = 0.70
MAX_CENTER_TRIES = 1000000
np.random.seed(RANDOM_SEED)

# ===================== FUNCTIONS =====================
    
def generate_cluster_sizes(clusterInfo: np.ndarray) -> Tuple[np.ndarray, int, float]:
    ci = np.asarray(clusterInfo, dtype=float)
    if ci.ndim == 1:
        ci = ci[None, :]
    if ci.shape[1] != 7:
        raise ValueError("clusterInfo must have 7 values: [count, baseX, deltaX, baseY, deltaY, baseZ, deltaZ]")
            
    all_dims = []
    for row in ci:
        count = int(round(row[0]))
        if count <= 0:
                continue
        baseX, deltaX, baseY, deltaY, baseZ, deltaZ = row[1:]
        sizeX = baseX + (np.random.rand(count, 1) * 2 * deltaX) - deltaX
        sizeY = baseY + (np.random.rand(count, 1) * 2 * deltaY) - deltaY
        sizeZ = baseZ + (np.random.rand(count, 1) * 2 * deltaZ) - deltaZ
        all_dims.append(np.hstack([sizeX, sizeY, sizeZ]))
                    
    clusterDims = np.vstack(all_dims) if all_dims else np.empty((0, 3), dtype=float)
    numClusters = clusterDims.shape[0]
    maxRadius = float(np.max(clusterDims)) if numClusters > 0 else 0.0
    return clusterDims, numClusters, maxRadius


def generateClusterCenters(xMin, xMax, yMin, yMax, zMin, zMax, numClusters, sepFactor, maxRadius):
    xMin *= 0.8; xMax *= 0.8
    yMin *= 0.8; yMax *= 0.8
    zMin *= 0.8; zMax *= 0.8
                            
    candidateCount = int(1e6)
    xLower, xUpper = xMin + maxRadius, xMax - maxRadius
    yLower, yUpper = yMin + maxRadius, yMax - maxRadius
    zLower, zUpper = zMin + maxRadius, zMax - maxRadius
                            
    if not (xLower < xUpper and yLower < yUpper and zLower < zUpper):
        raise ValueError("Invalid bounds after applying maxRadius buffer.")
                                
    randX = (xUpper - xLower) * np.random.rand(candidateCount) + xLower
    randY = (yUpper - yLower) * np.random.rand(candidateCount) + yLower
    randZ = (zUpper - zLower) * np.random.rand(candidateCount) + zLower
                                
    selectedX, selectedY, selectedZ = [randX[0]], [randY[0]], [randZ[0]]
    minAllowedDist = sepFactor * maxRadius
                                
    for i in range(1, candidateCount):
        dx = np.array(selectedX) - randX[i]
        dy = np.array(selectedY) - randY[i]
        dz = np.array(selectedZ) - randZ[i]
        distances = np.sqrt(dx*dx + dy*dy + dz*dz)
        if np.min(distances) >= minAllowedDist:
            selectedX.append(randX[i])
            selectedY.append(randY[i])
            selectedZ.append(randZ[i])
            if len(selectedX) >= numClusters:
                break
                                            
    if len(selectedX) < numClusters:
        raise RuntimeError("Not enough space for requested clusters.")
                                                
    idx = np.random.permutation(len(selectedX))[:numClusters]
    centers = np.column_stack([np.array(selectedX)[idx], np.array(selectedY)[idx], np.array(selectedZ)[idx]])
                                                
    minDistance = np.min(pdist(centers)) if len(centers) >= 2 else np.inf
    return centers, minDistance


def generate_cr_clusters(Centers, Radii, CrPercentage, TotalPoints):
    total_cr = int(round(CrPercentage * TotalPoints))
    n_clusters = Centers.shape[0]
    max_per_cluster = total_cr // n_clusters
    if max_per_cluster <= 0:
        return np.empty((0, 4)), 0
                                                        
    pts_list = []
    cr_used = 0
    for i in range(n_clusters):
        rx, ry, rz = Radii[i]
        num_points = max_per_cluster
        xyz = np.hstack([np.random.randn(num_points, 1) * rx + Centers[i, 0],
                         np.random.randn(num_points, 1) * ry + Centers[i, 1],
                         np.random.randn(num_points, 1) * rz + Centers[i, 2]])
        labels = np.full((num_points, 1), i + 1, dtype=float)
        pts_list.append(np.hstack([xyz, labels]))
        cr_used += num_points
                                                            
    return np.vstack(pts_list), cr_used
                                                            

def generateBackground(Dim, Centers, dmax, NumCr, NumFe):
    total = NumCr + NumFe
    pts = np.random.rand(total, 3) * Dim
    keep = np.ones(total, dtype=bool)
    for c in Centers:
        d = np.sqrt(np.sum((pts - c)**2, axis=1))
        keep &= (d > dmax)
    pts = pts[keep, :]
                                                                
    if pts.shape[0] < total:
        warnings.warn("Not enough background space. Reducing background atoms.")
        total = pts.shape[0]
        NumCr = int(round((NumCr / (NumCr + NumFe)) * total))
        NumFe = total - NumCr
                                                                    
    return pts[:NumCr, :], pts[NumCr:NumCr + NumFe, :]


# ===================== MAIN PIPELINE =====================
if __name__ == "__main__":
    print("Generating cluster sizes...")
    Radii, numClusters, maxRadius = generate_cluster_sizes(clusterInfo)

    print("Generating cluster centers...")
    Centers, minDist = generateClusterCenters(0, Dim[0], 0, Dim[1], 0, Dim[2],
    numClusters, MinClusterSeparation, maxRadius)
    print(f"Centers generated: {Centers.shape[0]} (min distance = {minDist:.2f} nm)")

    print("Generating Cr clusters...")
    CrPoints_clusters, cr_used = generate_cr_clusters(Centers, Radii, F_CLUSTERED * CrPercentage, TotalNumPoints)
    print(f"Cr points in clusters: {CrPoints_clusters.shape[0]}")

    remainingCr = TotalNumCr - cr_used
    print(f"Remaining Cr for background: {remainingCr}")

    print("Generating background points...")
    Cr_bg, Fe_bg = generateBackground(Dim, Centers, maxRadius, remainingCr, TotalNumFe)
    print(f"Background: Cr = {Cr_bg.shape[0]}, Fe = {Fe_bg.shape[0]}")

    # Combine data
    all_points = {
        'Cr_clusters': CrPoints_clusters,
        'Cr_background': Cr_bg,
        'Fe_background': Fe_bg,
        'Centers': Centers,
        'Radii': Radii,
        'Dim': Dim
    }

    # Save as .mat file
    output_file = os.path.join(OutputFolder, "synthetic_clusters.mat")
    savemat(output_file, all_points)
    print(f"Data saved to {output_file}")
    
    # SAve Cr_clusters seperately as a .txt file
    cr_clusters_file=os.path.join(OutputFolder,"Cr_clusters.txt")
    np.savetxt(cr_clusters_file,CrPoints_clusters,fmt='%.6f')

    # Optional: Visualization (sample only for speed)
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(CrPoints_clusters[:2000, 0], CrPoints_clusters[:2000, 1], CrPoints_clusters[:2000, 2], c='r', s=1, label='Cr Clusters')
#     ax.scatter(Cr_bg[:2000, 0], Cr_bg[:2000, 1], Cr_bg[:2000, 2], c='g', s=1, label='Cr Background')
#     ax.scatter(Fe_bg[:2000, 0], Fe_bg[:2000, 1], Fe_bg[:2000, 2], c='b', s=1, label='Fe Background')
#     ax.set_xlim(0, Dim[0]); ax.set_ylim(0, Dim[1]); ax.set_zlim(0, Dim[2])
#     ax.legend()
#     plt.title("Synthetic Atom Distribution (Sample)")
#     plt.show()





