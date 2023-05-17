from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger, ReduceLROnPlateau
import pandas as pd


# Load our images

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   horizontal_flip= True,
                                   rotation_range=30)
train_generator = train_datagen.flow_from_directory(
    'train_images2',
    target_size=(96,96),
    batch_size=64,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale = 1./255, horizontal_flip= True, rotation_range=30)
test_generator = test_datagen.flow_from_directory(
    'test_images2',
    target_size=(96,96),
    batch_size=64,
    class_mode='categorical')


# Architecture

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape=(96, 96, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(60, activation='softmax'))

# Compile model with suitable loss function and optimizer
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())
# Train model on training set
model.fit(train_generator, epochs=20, batch_size=64, validation_data=test_generator)
# History
# Define the file path to save the history

# Train model on training set
history = model.fit(train_generator, epochs=20, batch_size=64, validation_data=test_generator)
# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = 'history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)

# or save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
# Evaluate model on validation set
loss, acc = model.evaluate(test_generator)
print('Validation loss:', loss)
print('Validation accuracy:', acc)


# Save model
model.save('dogai3.h5')
print(history)