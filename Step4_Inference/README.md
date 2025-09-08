# Inference

Once the model has been trained, users can directly apply it to their own testing data for evaluation.
Three example images '19.jpg'-'21.jpg' are provided in the directory. You can replace them with your own images if needed. 

## Python Example Code

```python
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable

# Path to the folder containing .jpg images
image_folder = './'
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

# Load all images into a numpy array
X = np.array([
    np.array(Image.open(os.path.join(image_folder, fname)).resize((100, 100)))
    for fname in image_files
])
X = X / 255.0  # if your model expects normalized input

# Load the model and specify the custom objects
loaded_model = load_model('ConvTiny.keras') 
# OR
loaded_model = load_model('ResNet50.keras')

# Predict
predictions = loaded_model.predict(X)
print(predictions)

import scipy.io
scipy.io.savemat('predictions.mat', {'predictions': predictions})
```

