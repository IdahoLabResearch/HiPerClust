# Synthetic Data Subvolume Extraction and Image Generation (Python)
# ---------------------------------------------------------------

import os
import math
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib import colors
from scipy.io import loadmat


# ============================= USER INPUTS =============================
base_folder = "./SyntheticData"   # <-- change to your base folder
output_folder = "Train"                           # <-- folder to save generated images
num_synthetic_files = 1                           # <-- total number of synthetic datasets
cube_size = 20.0                                  # <-- subvolume size in nm
grid_size = 100                                   # <-- grid size for projection images
min_points_threshold = 300                        # <-- minimum number of points in subvolume to process
min_cluster_size = 100                            # <-- minimum cluster size to consider
resize_image_size = (100, 100)                    # <-- final image size (pixels) (width, height)

# ======================================================================
 
os.makedirs(output_folder, exist_ok=True)
 
iter_counter = 1
parameters = []          # to mirror MATLAB "parameter" cell array (stores cluster_num)
groundtruth = []         # rows: [syn, i, j, k, cluster_num]
 
def count_rgb(img: np.ndarray, out_size=(100, 100)) -> Image.Image:
    # """
    # Emulates MATLAB's imagesc + current colormap -> RGB, then resizes.
    # - Uses matplotlib's 'viridis' by default (stable, perceptually uniform).
    # - Scales to [0, max(img)], handling the all-zero case.
    # """
    img = np.asarray(img, dtype=float)
    vmax = float(img.max()) if img.size > 0 else 0.0
    if vmax <= 0.0:
        # All zeros -> produce a black image
        rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.float32)
    else:
        norm = colors.Normalize(vmin=0.0, vmax=vmax, clip=True)
        cmap = cm.get_cmap("viridis")  # pick a fixed, nice colormap
        rgba = cmap(norm(img))         # floats in [0,1], shape (H,W,4)
 
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(rgb, mode="RGB")
    # MATLAB imresize default is bicubic-like; Pillow.BILINEAR is a reasonable match
    return pil_img.resize(out_size, resample=Image.BILINEAR)


for syn in range(1, num_synthetic_files + 1):
#     folder_name = f"GeneratedDatasetResults_Synthetic{syn}"
#     file_name = f"Synthetic{syn}_CuAtomsInClusterandSolidSolution.txt"
   file_name='Cr_clusters.txt'
   path = os.path.join(base_folder, file_name)


   # Read the dataset (expecting numeric columns; 5th col is label)
   # If your files have headers or different delimiters, adjust loadtxt arguments.
   Data = np.loadtxt(path)
 

   # Work only with spatial columns for cube tiling
   mins = Data[:, :3].min(axis=0)
   maxs = Data[:, :3].max(axis=0)
   spans = maxs - mins

   # Number of cubes along x, y, z
   num_cubes = np.floor(spans / cube_size).astype(int)  # shape (3,)

   # Guard: if any dimension has 0 cubes, nothing to do
   if np.any(num_cubes <= 0):
       continue

   # Divide dataset into subvolumes 
   for i in range(1, num_cubes[0] + 1):
       x0 = mins[0] + cube_size * (i - 1)
       x1 = x0 + cube_size
       in_x = (Data[:, 0] >= x0) & (Data[:, 0] <= x1)

       for j in range(1, num_cubes[1] + 1):
           y0 = mins[1] + cube_size * (j - 1)
           y1 = y0 + cube_size
           in_y = (Data[:, 1] >= y0) & (Data[:, 1] <= y1)

           for k in range(1, num_cubes[2] + 1):
               z0 = mins[2] + cube_size * (k - 1)
               z1 = z0 + cube_size
               in_z = (Data[:, 2] >= z0) & (Data[:, 2] <= z1)

               gt = Data[in_x & in_y & in_z]

               # Remove invalid points (label = -1) — assumes labels are in column 5 (0-based idx 4)
               if gt.size == 0:
                   continue
               gt = gt[gt[:, 2] != -1]

               if gt.shape[0] <= min_points_threshold:
                   continue

               # Count clusters >= min_cluster_size
               labels = gt[:, 3].astype(int)
               # Equivalent to MATLAB: [uniqueVals, ~, idx] + accumarray
               unique_vals, counts = np.unique(labels, return_counts=True)
               t_filtered = counts[counts >= min_cluster_size]
               cluster_num = int(t_filtered.size)

               # Scale coordinates to [0, grid_size - 1]
               sub_min = gt[:, :3].min(axis=0)
               sub_max = gt[:, :3].max(axis=0)
               ranges = sub_max - sub_min
               # avoid division by zero for flat ranges
               ranges[ranges == 0] = 1.0
               scaled = (gt[:, :3] - sub_min) / ranges * (grid_size - 1)

               # Integer indices (MATLAB used floor + 1; here we keep 0-based)
               idxs = np.floor(scaled).astype(int)
               # clamp just in case of numerical edge (e.g., exactly grid_size)
               np.clip(idxs, 0, grid_size - 1, out=idxs)

               # Initialize projections
               Count_xy = np.zeros((grid_size, grid_size), dtype=np.int32)
               Count_xz = np.zeros((grid_size, grid_size), dtype=np.int32)
               Count_yz = np.zeros((grid_size, grid_size), dtype=np.int32)

               # Populate projections (vectorized with np.add.at)
               ix, iy, iz = idxs[:, 0], idxs[:, 1], idxs[:, 2]
               np.add.at(Count_xy, (ix, iy), 1)
               np.add.at(Count_xz, (ix, iz), 1)
               np.add.at(Count_yz, (iy, iz), 1)

               # Save images (three views)
               for mat in (Count_xy, Count_xz, Count_yz):
                   img = count_rgb(mat, out_size=resize_image_size)
                   out_path = os.path.join(output_folder, f"{iter_counter}.jpg")
                   img.save(out_path, format="JPEG", quality=95)

                   parameters.append([cluster_num])
                   groundtruth.append([syn, i, j, k, cluster_num])
                   iter_counter += 1
               

# (Optional) Save metadata like MATLAB variables would
# np.savetxt(os.path.join(output_folder, "parameters.csv"), np.array(parameters, dtype=int), fmt="%d", delimiter=",")
# np.savetxt(os.path.join(output_folder, "groundtruth.csv"), np.array(groundtruth, dtype=int), fmt="%d", delimiter=",")

print(f"Done. Wrote {iter_counter-1} images to: {output_folder}")




