import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pathlib
import os
import PIL.Image
from keras.src.applications import EfficientNetB4
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import math
import keras
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans
from collections import Counter


def config(mode):
    i_size = 380
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
    model = EfficientNetB4(weights='imagenet')
    label_predictions = {}

    for images, labels in train_ds:
        x = keras.applications.efficientnet.preprocess_input(images)
        preds = model.predict(x)
        for i in range(len(preds)):
            label = class_names[labels[i].numpy()]  # Get the class name corresponding to the label
            decoded_preds = keras.applications.efficientnet.decode_predictions(preds, top=3)[i]
            top_prediction = decoded_preds[0][1]  # Get the class name of the top prediction
            if label not in label_predictions:
                label_predictions[label] = []
            label_predictions[label].append(top_prediction)

    # Convert the dictionary to a list of tuples and sort it
    sorted_label_predictions = sorted(label_predictions.items(), key=lambda item: item[0])

    for label, predictions in sorted_label_predictions:
        counter = Counter(predictions)
        total_predictions = len(predictions)
        most_common_predictions = counter.most_common(3)  # Get the two most common predictions
        most_common_predictions_percentages = [(pred[0], round(pred[1] / total_predictions, 2)) for pred in most_common_predictions]
        print(f'{label}: {most_common_predictions_percentages}')

    return None


def create_model():
    resize_and_rescale = keras.Sequential([
        keras.layers.Resizing(150, 150),
        keras.layers.Rescaling(1. / 255)
    ])

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.2),
    ])

    model = keras.Sequential()
    model.add(resize_and_rescale)
    model.add(data_augmentation)
    model.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Conv2D(filters=8, kernel_size=2, activation='relu'))
    model.add(keras.layers.MaxPooling2D())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(90, activation='softmax'))
    return model


def train_convolutions():
    model = create_model()
    optimizer = keras.optimizers.Adam(learning_rate=0.00015)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_ds, epochs=40, validation_data=val_ds, callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)])

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

    print(classification_report(actuals_train, predictions_train))
    print(classification_report(actuals_test, predictions_test))

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


def create_transfer_model():
    preprocess_input = keras.applications.efficientnet.preprocess_input
    data_augmentation = keras.Sequential([
        keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    base_model = EfficientNetB4(weights='imagenet', include_top=False)
    base_model.trainable = False
    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(90, activation='softmax')

    inputs = keras.Input(shape=(300, 300, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs, outputs)
    return model


def train_transfer_model():
    model = create_transfer_model()
    base_learning_rate = 0.0001
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=base_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(train_ds, epochs=40, validation_data=val_ds, callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)])

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

    print(classification_report(actuals_train, predictions_train))
    print(classification_report(actuals_test, predictions_test))

    return None


def createDataset():
    # Load the EfficientNetB3 network, ensuring the head FC layer sets are left off
    model = EfficientNetB4(weights="imagenet", include_top=False)

    # Initialize our dataframe
    df = pd.DataFrame(columns=["pathOfImage", "actualClass"] + [f"feature_{i}" for i in range(model.output_shape[-1])])

    # List of directories to iterate over
    directories = [os.path.join(base_dir, 'train'), os.path.join(base_dir, 'test')]

    for directory in directories:
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):  # Check if the path is a directory
                for filename in os.listdir(folder_path):
                    # Load and preprocess the image
                    img_path = os.path.join(folder_path, filename).replace("\\", "/")
                    img = keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
                    img_path = "data" + img_path.split("data", 1)[1]
                    x = keras.preprocessing.image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = keras.applications.efficientnet.preprocess_input(x)

                    # Generate features
                    features = model.predict(x)

                    # Flatten the 2D array to a 1D array
                    features = features.flatten()

                    # Save the image path, actual class, and features to the dataframe
                    data = {"pathOfImage": img_path, "actualClass": os.path.basename(folder)}
                    data.update({f"feature_{i}": feature for i, feature in enumerate(features)})
                    df.loc[len(df)] = data

    # Save the dataframe to a CSV file
    df.to_csv("dataset1.csv", index=False)
    return df


def clusterDataset():
    # Load the dataset
    df = pd.read_csv("dataset1.csv")

    # Extract the features from the dataframe
    features = df[[f"feature_{i}" for i in range(1792)]].values

    pca = PCA(n_components=5, random_state=71)
    features_pca = pca.fit_transform(features)

    kmeans = KMeans(n_clusters=35, n_init='auto', random_state=71).fit(features_pca)

    df['cluster'] = kmeans.labels_

    # Save the dataframe to a CSV file
    df.to_csv("dataset2.csv", index=False)
    return df


def show_cluster_images():
    df = pd.read_csv("dataset2.csv")
    clusters = df['cluster'].unique()

    # Print the number of units in each cluster
    cluster_counts = df['cluster'].value_counts()
    print(cluster_counts)

    for cluster in clusters:
        # Get the maximum of 25 or the total number of entries in the current cluster
        sample_size = min(30, df[df['cluster'] == cluster].shape[0])
        cluster_images = df[df['cluster'] == cluster].sample(sample_size)

        if df[df['cluster'] == cluster].shape[0] > 10:

            fig = plt.figure(figsize=(15, 15))  # Define the figure size

            for i, row in enumerate(cluster_images.iterrows()):
                _, row_data = row
                im = PIL.Image.open(row_data['pathOfImage'])

                ax = fig.add_subplot(5, 6, i + 1)  # Add a subplot for each image
                ax.imshow(im)
                ax.axis('off')  # Hide axes
                ax.set_title(f'Cluster {cluster}')

            plt.show()
    return None


def show_average_images():
    df = pd.read_csv("dataset2.csv")
    clusters = df['cluster'].unique()

    # Calculate the grid size based on the number of clusters
    grid_size = math.ceil(math.sqrt(len(clusters)))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))

    for i, cluster in enumerate(clusters):
        # Get all images in the current cluster
        cluster_images = df[df['cluster'] == cluster]['pathOfImage']

        # Initialize an empty list to store the images
        images = []
        # Read each image, convert to numpy array and append to the list
        for image_path in cluster_images:
            im = PIL.Image.open(image_path)
            im = im.resize((50, 50))  # Resize the image
            im_array = np.array(im)
            images.append(im_array)

        # Calculate the average image
        avg_image = np.mean(images, axis=0)

        # Display the average image in a subplot
        ax = axs[i // grid_size, i % grid_size]
        ax.imshow(avg_image.astype(np.uint8))
        ax.axis('off')  # Hide axes
        ax.set_title(f'Cluster {cluster}')

    # Remove unused subplots
    for j in range(i + 1, grid_size * grid_size):
        fig.delaxes(axs.flatten()[j])

    plt.tight_layout()
    plt.show()

    return None


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
# test_imagenet_model_on_test_data()
# train_convolutions()
# train_transfer_model()
# createDataset()
clusterDataset()
show_cluster_images()
show_average_images()
