import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import os
import PIL.Image
from keras.src.applications import ResNet50
from keras.src.applications.convnext import preprocess_input, decode_predictions
from sklearn.metrics import confusion_matrix
import math
import keras

# base_dir = 'C:/Users/koros/Desktop/SUNS/Zadania/Zadanie3/SUNS-Zadanie3/data'
base_dir = 'D:/Skola/ING/SEM1/SUNS/SUNS-Zadanie3/data'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
animals_folders = list(pathlib.Path(train_dir).glob('*'))

img_size = 224
batch_size = 32


def initialize_data():
    print("Train data: ")
    train_das = keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        validation_split=0.1,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)

    print("Validation data: ")
    val_das = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.1,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)

    print("Test data: ")
    test_das = keras.utils.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)

    print("Class names: ")
    class_names = train_das.class_names
    print(class_names)

    return train_das, val_das, test_das


def show90animals():
    # Calculate the grid size based on the number of animal folders
    grid_size = math.ceil(math.sqrt(len(animals_folders)))

    fig = plt.figure(figsize=(10, 10))  # Define the figure size

    for i, animal_folder in enumerate(animals_folders):
        animal_images = list(animal_folder.glob('*'))
        im = PIL.Image.open(str(animal_images[0]))

        ax = fig.add_subplot(grid_size, grid_size, i + 1)  # Add a subplot for each image
        ax.imshow(im)
        ax.axis('off')  # Hide axes

    plt.show()
    return None


def test_imagenet_model_on_test_data():
    model = ResNet50(weights='imagenet')
    
    return None


train_ds, val_ds, test_ds = initialize_data()
show90animals()
