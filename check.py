import os

# Path to the training dataset directory (can be train, test, or valid folder)
train_dir =r'E:\VScode\Dataset\archive\train'  # Modify this to the correct path

# List the directories (class labels) in the 'train' folder
class_labels = sorted(os.listdir(train_dir))

# Create a dictionary mapping class labels to numeric indices
class_labels_dict = {i: label for i, label in enumerate(class_labels)}

# Print the class labels dictionary
print("Class labels:", class_labels_dict)

# Now you can use this `class_labels_dict` in your code

print("Extracted Class Labels:", class_labels_dict)
