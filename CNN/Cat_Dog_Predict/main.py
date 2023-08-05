import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 12
EPOCHS = 20
NUM_CLASSES = 2  # Cat and Dog

# Paths to the dataset
train_data_dir = 'data/train'
test_data_dir = 'data/validation'


# ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0/255,       # Rescale pixel values to [0, 1]
    shear_range=0.2,       # Random shear augmentation
    zoom_range=0.2,        # Random zoom augmentation
    horizontal_flip=True   # Random horizontal flip augmentation
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load and preprocess training data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load and preprocess test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Create the CNN model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=test_generator,
    validation_steps=len(test_generator)
)

model.save("cat_dog_classifier.h5")
print("Saved model...")

if __name__ == "__main__":
    print(train_data_dir)
    print(train_generator)
    print(test_generator)