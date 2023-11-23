import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import os
import PIL.Image
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# base_dir = 'C:/Users/koros/Desktop/SUNS/Zadania/Zadanie3/SUNS-Zadanie3/data'
base_dir = 'C:/Users/koros/Desktop/SUNS/Zadania/Zadanie3/SUNS-Zadanie3/data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
animals_folders = list(pathlib.Path(train_dir).glob('*'))

img_size = 224
batch_size = 32

def initialize_data(i_size, b_size, train_d, test_d):
    train_ds = tf.keras.utils.image_dataset_from_directory(
    train_d,
    shuffle=True,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(i_size, i_size),
    batch_size=b_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    train_d,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(i_size, i_size),
    batch_size=b_size)

    test_ds = tf.keras.utils.image_dataset_from_directory(
    test_d,
    seed=123,
    image_size=(i_size, i_size),
    batch_size=b_size)

    class_names = train_ds.class_names
    print(class_names)

    return train_ds, val_ds, test_ds

def show90animals(anim_folders):
  # Calculate the grid size based on the number of animal folders
  grid_size = math.ceil(math.sqrt(len(anim_folders)))

  fig = plt.figure(figsize=(10, 10))  # Define the figure size

  for i, animal_folder in enumerate(anim_folders):
      animal_images = list(animal_folder.glob('*'))
      im = PIL.Image.open(str(animal_images[0]))
      
      ax = fig.add_subplot(grid_size, grid_size, i+1)  # Add a subplot for each image
      ax.imshow(im)
      ax.axis('off')  # Hide axes

  plt.show()
  return None


def imagenet_prediction(test_ds):
    # Load the pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')

    # Initialize lists to store the predictions and true labels
    y_pred = []
    y_true = []

    # Iterate over the test dataset
    for images, labels in test_ds:
        # Preprocess the images
        images = preprocess_input(images)

        # Make predictions
        preds = model.predict(images)

        # Decode the predictions
        decoded_preds = decode_predictions(preds, top=1)

        # Append the predictions and true labels to the lists
        y_pred.extend(decoded_preds)
        y_true.extend(labels.numpy())


    return y_true, y_pred
  
train_ds, val_ds, test_ds = initialize_data(img_size, batch_size, train_dir, test_dir)
initialize_data(img_size, batch_size, train_dir, test_dir)
imagenet_prediction(test_ds)