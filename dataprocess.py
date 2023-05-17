import cv2
from dataread import df_train, df_labels
import os


if not os.path.exists("cropped_images2"):
    os.mkdir("cropped_images2")

def preprocess(image_path, xmin, ymin, xmax, ymax, target_size):

    # Load image
    image = cv2.imread(image_path)

    # Crop the image
    crop_img = image[ymin:ymax, xmin:xmax]

    # Image resize
    resized_img = cv2.resize(crop_img, target_size)

    normalized_img = resized_img / 255.0

    return normalized_img


# Size
target_size = (96,96)

# Splice data and handle cropping
for index, row in df_train.iterrows():
    breed = row['breed']
    image_path = row['image_path']
    xmin = row['xmin']
    ymin = row['ymin']
    xmax = row['xmax']
    ymax = row['ymax']


    img = preprocess(image_path, xmin, ymin, xmax, ymax, target_size)

    # Save image
    save_path = os.path.join("cropped_images2", os.path.dirname(image_path)[14:])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, os.path.basename(image_path))
    cv2.imwrite(save_path, img * 255.0)
    