from keras.models import load_model
from dataprocess import preprocess
import numpy as np

model = load_model('dogai.h5')

image_path = 'train_images/n02085620-Chihuahua/n02085620_242.jpg'
image  = preprocess(image_path, 113,45,368,486, (96,96))

image = np.expand_dims(image, axis = 0)

predictions = model.predict(image)

predicted_class = np.argmax(predictions)

# Print the predicted class label
print("Predicted class:", predicted_class)