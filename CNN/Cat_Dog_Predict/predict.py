import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

# Constants
IMG_WIDTH, IMG_HEIGHT = 150, 150
CLASS_LABELS = ['cat', 'dog']

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image_path):
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    class_label = CLASS_LABELS[class_index]
    return class_label

def main():
    # Load the pre-trained model
    model = load_model('cat_dog_classifier.h5')

    # Test image path
    test_image_path = 'data/train/dogs/dog.11.jpg'

    if os.path.exists(test_image_path):
        class_label =  predict_image(model, test_image_path)
        print(f"Prediction: {class_label}")
    else:
        print("Error: Test image file not found.")

if __name__ == "__main__":
    main()
