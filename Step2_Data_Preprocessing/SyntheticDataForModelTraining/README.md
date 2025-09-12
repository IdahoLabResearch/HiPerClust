# Synthetic Datasets for Model Training

This folder contains 630 synthetic datasets (.mat files) intended to be preprocessed through Step_2_Data_Preprocessing before they can be used for Step3_Model_training.

## 📂 File Contents
Each `synthetic_*_img.mat` file contains several variables, with the following being the most important:  
1. `Data`  
* Format: [x, y, z, elementID, label]  
    - x, y, z: Atom coordinates.  
    - elementID: Atomic species (not relevant for - this project, can be ignored).  
    - label: Cluster assignment (label=-1, noise points; label=1-N, Cluster labels).

2. `data` (cell array)  
* `data` is an i × j × k cell array created by partitioning the `Data` into subvolumes (based on `cube_size`).  
* Each subvolume will be projected onto the XY, XZ, and YZ planes to generate RGB images. These images are saved in the user-defined output folder during preprocessing.

3. `groundtruth`  
* Format: [syn, i, j, k, cluster_num]  
    - syn: Index of the synthetic dataset (e.g., syn=1,synthetic_1_img.mat).  
    - i, j, k: Subvolume indices within the partitioned data.  
    - cluster_num: Number of clusters contained in this subvolume.
* Example:  
    `groundtruth` = [1, 2, 1, 2, 4]  
    - Generated from synthetic_1_img.mat.  
    - Subvolume located at data{2,1,2}.  
    - Contains 4 clusters.

4. Other Variables
* Additional intermediate variables are created during preprocessing can be ignored for model training.

## 🔑 Notes 
* Always preprocess the .mat files with the scripts in Step2_Data_Preprocessing before proceeding to Step 3.

* The preprocessing step automatically generates both:
RGB projection images.  
Ground-truth cluster labels.

* Ensure your folder paths and cube_size settings are correctly defined in the preprocessing script.
