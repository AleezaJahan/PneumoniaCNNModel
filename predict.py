import tensorflow as tf
import numpy as np
import cv2
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

img_width, img_height = 128, 128

# Parsing arguments for the script
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
                help="path to the chest X-ray image")
args = vars(ap.parse_args())

# Load the pre-trained model
model = load_model('model.h5')

# Load, preprocess, and predict on the image
image = cv2.imread(args["image"])
image = cv2.resize(image, (img_width, img_height))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

result = model.predict(image)
prediction = "Normal" if result[0][0] < 0.5 else "Pneumonia"  # Assuming 'Normal' is the class associated with the lower probability

print(f"The prediction is: {prediction}")
