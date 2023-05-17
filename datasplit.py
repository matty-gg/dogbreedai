import pandas as pd
import cv2
import os

# Read train and test set CSV files
df_train = pd.read_csv("train_set2.csv")
df_test = pd.read_csv("test_set2.csv")

# Define train and test image folders
train_folder = "train_images2"
test_folder = "test_images2"

# Loop over each row in the train set dataframe
for index, row in df_train.iterrows():
    breed = row['breed']
    image_path = row['image_path']

    # Read the image using cv2
    image = cv2.imread(image_path)

    # Define the save path for the image
    save_path = os.path.join(train_folder, os.path.dirname(image_path)[16:])

    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Append the filename to the save path and save the image
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_path, filename)
    cv2.imwrite(save_path, image)

# Loop over each row in the test set dataframe
for index, row in df_test.iterrows():
    breed = row['breed']
    image_path = row['image_path']

    # Read the image using cv2
    image = cv2.imread(image_path)

    # Define the save path for the image
    save_path = os.path.join(test_folder, os.path.dirname(image_path)[16:])

    # Create the directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Append the filename to the save path and save the image
    filename = os.path.basename(image_path)
    save_path = os.path.join(save_path, filename)
    cv2.imwrite(save_path, image)