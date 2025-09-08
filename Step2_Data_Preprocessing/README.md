# 2D Projection Image Generation from 3D Atomic Data

This module generates 2D projection images from 3D synthetic APT atomic datasets. These images can be used for:
-  deep learning model training
- clustering validation
- data visualization.

## 1. What is a projection?

To prepare the data for model training and evaluation, three dimensional APT datasets were transformed into two-dimensional density projection images. It takes 3D raw APT data as input, regardless of the data volume (measured in nm), and divides the entire space into subvolumes of size (such as 20 × 20 × 20, nm3). For each subvolume, the 3D spatial point data are projected onto the XY, XZ, and YZ planes, resulting in 2D images that represent the distribution of atoms from different perspectives. These projections visually highlight dense regions, where the number of distinct colored areas corresponds to the number of clusters in the original 3D space. The color intensity in each image was normalized to the maximum local density within the subvolume, with lighter colors indicating regions of higher atomic concentration. Each image was then resized to 100 × 100 pixels to ensure uniform input dimensions for the neural network.

A projection is a way to represent 3D data in 2D by looking along one axis:

- XY projection: Looking from the Z-axis (top view).

- XZ projection: Looking from the Y-axis (front view).

- YZ projection: Looking from the X-axis (side view).

So, each 3D subvolume will generate three 2D projection images.

<p align="center">
    <img src="Images/projection.png" alt="Image" Width="80%">
    <br>
    <em> Figure 1: Three-dimensional point cloud representation of atom probe tomography (APT) data, alongside its 2D projections onto the xy, xz, and yz planes. The projected density shadows qualitatively illustrate the spatial distribution and potential number of atomic clusters present in the dataset. </em>
</p>

## 2. Code Overview

The main steps:

    1. Load datasets from basePath.

    2. Split into subvolumes (cubes of size cubeSize).

    3. Apply threshold filters to ensure enough points exist in each cube.

    4. Generate projections for XY, XZ, YZ planes.

    5. Resize and save images in the saveFolder (default: Train).

    6. Save ground truth labels in groundtruth.mat.  

## 3. User Input Parameters

| Parameter           | Description                                         | Typical Value           |
|---------------------|-----------------------------------------------------|--------------------------|
| basePath            | Folder containing synthetic 3D datasets             | `./synthetic_data/`     |
| numSyntheticFiles   | Number of synthetic files to process                | 630                   |
| cubeSize            | Size of each subvolume in nanometers                | 20                    |
| gridsize            | Grid size for projection image (before resizing)    | 100                   |
| saveFolder          | Output folder for images                            | `Train`                 |
| minPointsThreshold  | Minimum number of points in a cube                  | 300                   |
| minClusterSize      | Minimum cluster size for ground truth labeling      | 100                   |
| imageSize           | Final image size in pixels (for CNN input)          | [100, 100]            |

## 4. How to choose reasonable values

- cubeSize: The cubeSize or subvolume sizes for the data should be tailored to highlight typical clustering behaviors in the alloy. This subvolume size was chosen to balance the inclusion of
a sufficient number of clusters for model learning, without overwhelming the projection with excessive overlap. Smaller cubes = more images, but less context. Typical: 10–30 nm.

- gridsize: Affects resolution before resizing. Keep ≥100 for good detail.

- imageSize: Match your CNN input size.

- minPointsThreshold: Avoid empty projections. Adjust based on dataset density.

- minClusterSize: Ignore tiny clusters to reduce noise in labels.

## 5. Customization Tips

- Images: .jpg format, 3 projections per cube.

- Ground Truth: groundtruth.mat with:

        - groundtruth: [dataset_id, i, j, k, cluster_num]

        - parameter: Total clusters per projection.
