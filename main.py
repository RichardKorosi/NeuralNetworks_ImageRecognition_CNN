import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pathlib
import os
import PIL.Image
from keras.src.applications import ResNet50
from keras.src.applications.convnext import preprocess_input, decode_predictions
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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


train_ds, val_ds, test_ds = initialize_data()


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


def create_model():
    return keras.models.Sequential([
        keras.layers.Flatten(input_shape=(224, 224, 3), name='layers_flatten'),
        keras.layers.Dense(512, activation='relu', name='layers_dense'),
        keras.layers.Dropout(0.2, name='layers_dropout'),
        keras.layers.Dense(90, activation='softmax', name='layers_dense_2')
    ])


def train_convolutions():
    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_ds, epochs=10, validation_data=val_ds)

    # Get the model's predictions on the validation dataset
    predictions = model.predict(val_ds)
    predicted_classes = np.argmax(predictions, axis=1)

    # Get the true labels of the validation dataset
    true_labels = val_ds.classes

    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_classes)

    # Get class names
    class_names = val_ds.class_names

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion matrix on validation set")
    disp.ax_.set(xlabel='Predicted', ylabel='True')
    plt.show()

    return None


show90animals()
train_convolutions()
