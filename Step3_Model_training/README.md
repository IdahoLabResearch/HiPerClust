# Transfer Learning on ConvNeXt-Tiny and ResNet-50

## 1. Why transfer learning?

In the context of APT data/image analysis, due to the relatively lack of large labeled data sets for training and testing, we employ transfer learning to overcome the the problem of relatively small datasets. This approach commonly employs pretrained models initially trained on large-scale datasets such as ImageNet as feature extractors for related tasks. Specifically, we re-trained two large convolutional neural networks ConvNext-Tiny and ResNet-50 as feature extractor, originally trained with over million annotated images, to detect the number of clusters from synthetic APT data. Our hypothesis is that the feature extraction layers of ConvNeXt-Tiny and ResNet-50 will also perform well on cluster detection of APT cluster images.

The 2D projection images were passed through two models ResNet-50 and ConvNeXt-Tiny to predict the number of clusters per subvolume. These models, trained via transfer learning, performed regression to provide an estimate for the number of clusters, which informed subsequent clustering steps.

## 2. Code Overview

This folder contains the following files:

- ConvTiny.ipynb

    - Implements transfer learning using the ConvNeXtTiny model.

    - Includes data preprocessing, augmentation, model fine-tuning, and evaluation.

    - Saves training history and final model.

- ResNet50.ipynb

    - Implements transfer learning using the ResNet50 model.
    
    - Includes data preprocessing, augmentation, model fine-tuning, and evaluation.

    - Saves training history and final model.

- ConvTiny.keras & ResNet50.keras

    - Saved trained models for reuse or deployment.

## 3. Run training

An example training dataset (Train.mat) is provided in the directory. You can replace it with your own data if needed.

- Open ConvTiny.ipynb or ResNet50.ipynb in Jupyter Notebook.

- Modify file paths for .mat files if necessary.

- Run all cells to:

    1. Input Data

    - The dataset contains 2D images projected from 3D point clouds.

    - Input images are reshaped and augmented for training.

    2. Preprocessing

    - Normalization of data

    - Data augmentation (rotation, flipping, small shifts)

    3. Model Architecture

    - Base model: ConvNeXtTiny or ResNet50 (pretrained on ImageNet)

    - Add custom Dense layers, Dropout, and regularization

    - Fine-tune the base model (with unfreezing)

    4. Training

    - Loss: Mean Squared Error (MSE)

    - Metrics: Mean Absolute Error (MAE)

    - Early stopping & learning rate scheduler

    5. Evaluation & Prediction

    - Save model (.keras) and training history (.mat)

    - Generate predictions for test set


## 4. Outputs

    Trained models: .keras format

    Training history: .mat file (loss and metrics per epoch)

    Loss plots: PNG file (Model Loss and Accuracy)  
