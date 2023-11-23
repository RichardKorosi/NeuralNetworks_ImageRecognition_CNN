import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import os
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt
import math

img_size = 150
batch_size = 32

# Notebook
base_dir = 'C:/Users/koros/Desktop/SUNS/Zadania/Zadanie3/SUNS-Zadanie3/data'

# PC
# base_dir = 'C:/Users/koros/Desktop/SUNS/Zadania/Zadanie3/SUNS-Zadanie3/data'

train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

animals_folders = list(pathlib.Path(train_dir).glob('*'))

# Calculate the grid size based on the number of animal folders
grid_size = math.ceil(math.sqrt(len(animals_folders)))

fig = plt.figure(figsize=(10, 10))  # Define the figure size

for i, animal_folder in enumerate(animals_folders):
    animal_images = list(animal_folder.glob('*'))
    im = PIL.Image.open(str(animal_images[0]))
    
    ax = fig.add_subplot(grid_size, grid_size, i+1)  # Add a subplot for each image
    ax.imshow(im)
    ax.axis('off')  # Hide axes

plt.show()

train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  shuffle=True,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_size, img_size),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)
