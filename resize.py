import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import numpy as np

# Path to your input image and output folder
input_image_path = r"E:\VScode\Dataset\test\AppleCedarRust1.JPG"  # replace with your actual input image path
output_folder = r"E:\VScode\Resize"  # output folder

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load the image
img = Image.open(input_image_path)

# Resize the image to match the model input size (224x224)
img_resized = img.resize((224, 224))

# Convert image to numpy array and normalize the pixel values to [0, 1]
img_array = np.array(img_resized) / 255.0

# If the image is grayscale, convert it to RGB (3 channels)
if img_array.shape[-1] == 1:  # Grayscale image
    img_array = np.repeat(img_array, 3, axis=-1)  # Convert to RGB

# Check the shape of the image after resizing
print(f"Shape of preprocessed image: {img_array.shape}")

# Save the processed image to the output folder
output_image_path = os.path.join(output_folder, "processed_image.jpg")
img_resized.save(output_image_path)

print(f"Processed image saved to {output_image_path}")

# Optionally, you can also save the image array if needed
np.save(os.path.join(output_folder, "processed_image.npy"), img_array)

# Now, the image is ready to be passed into your model
