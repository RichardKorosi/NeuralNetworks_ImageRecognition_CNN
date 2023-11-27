import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pathlib
import os
import PIL.Image
from keras.src.applications import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.metrics import confusion_matrix
import math
import keras
import tensorflow as tf


def config(mode):
    i_size = 224
    b_size = 32

    if mode == 'notebook':
        base_directory = 'C:/Users/koros/Desktop/SUNS/Zadania/Zadanie3/SUNS-Zadanie3/data'
    elif mode == 'desktop':
        base_directory = 'D:/Skola/ING/SEM1/SUNS/SUNS-Zadanie3/data'
    else:
        base_directory = '/content/drive/MyDrive/data'
    train_directory = os.path.join(base_directory, 'train')
    test_directory = os.path.join(base_directory, 'test')
    animal_folders = list(pathlib.Path(train_directory).glob('*'))

    return i_size, b_size, base_directory, train_directory, test_directory, animal_folders


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
    classes_names = train_das.class_names
    print(classes_names)

    return train_das, val_das, test_das, classes_names


def show90animals():
    # Calculate the grid size based on the number of animal folders
    grid_size = math.ceil(math.sqrt(len(animals_folders)))

    fig = plt.figure(figsize=(15, 15))  # Define the figure size

    for i, animal_folder in enumerate(animals_folders):
        animal_images = list(animal_folder.glob('*'))
        im = PIL.Image.open(str(animal_images[0]))

        ax = fig.add_subplot(grid_size, grid_size, i + 1)  # Add a subplot for each image
        ax.imshow(im)
        ax.axis('off')  # Hide axes
        ax.set_title(class_names[i])

    plt.show()
    return None


def test_imagenet_model_on_test_data():
    model = ResNet50(weights='imagenet')
    predictions = {}

    for images, labels in test_ds:
        x = preprocess_input(images)
        preds = model.predict(x)
        for i in range(len(preds)):
            predictions[labels[i].numpy()] = decode_predictions(preds, top=3)[i]

    sorted_predictions = sorted(predictions.items(), key=lambda item: item[0])

    for label, prediction in sorted_predictions:
        class_name_and_prob = [(pred[1], pred[2]) for pred in prediction]
        print('Most predicted classes for ' + f'[{label}]' + (class_names[label]) + ':', class_name_and_prob)

    return None


def create_model():
    model = keras.Sequential()
    model.add(keras.layers.Rescaling(1. / 255, input_shape=(img_size, img_size, 3)))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(90, activation='softmax'))
    return model


def train_convolutions():
    model = create_model()
    optimizer = keras.optimizers.Adam(learning_rate=0.0002)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_ds, epochs=40, validation_data=val_ds, callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, restore_best_weights=True)])

    train_scores = model.evaluate(train_ds, verbose=0)
    test_scores = model.evaluate(test_ds, verbose=0)
    print(f"Train accuracy: {train_scores[1]:.4f}")
    print(f"Test accuracy: {test_scores[1]:.4f}")

    # Predictions and Confusion Matrix on Train Set
    predictions_train = model.predict(train_ds)
    predictions_train = np.argmax(predictions_train, axis=1)
    actuals_train = np.concatenate([y for x, y in train_ds], axis=0)
    cm_train = confusion_matrix(actuals_train, predictions_train)
    plot_confusion_matrix(cm_train, title="Confusion matrix on train set")

    # Predictions and Confusion Matrix on Test Set
    predictions_test = model.predict(test_ds)
    predictions_test = np.argmax(predictions_test, axis=1)
    actuals_test = np.concatenate([y for x, y in test_ds], axis=0)
    cm_test = confusion_matrix(actuals_test, predictions_test)
    plot_confusion_matrix(cm_test, title="Confusion matrix on test set")

    plotHistory(history)

    return None


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(33, 13))
    sns.heatmap(cm, annot=False, cmap='viridis', xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=20)
    plt.xticks(rotation=60)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)
    plt.show()


def plotHistory(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy/Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss/Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Results -------------------------------------------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

img_size, batch_size, base_dir, train_dir, test_dir, animals_folders = config('notebook')
# img_size, batch_size, base_dir, train_dir, test_dir, animals_folders = config('desktop')
# img_size, batch_size, base_dir, train_dir, test_dir, animals_folders = config('colab')
train_ds, val_ds, test_ds, class_names = initialize_data()

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# show90animals()
test_imagenet_model_on_test_data()
# train_convolutions()
