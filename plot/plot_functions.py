import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import PIL
import os


def plot_sample_images(data, path_to_images):
    fig = plt.figure(figsize=(12, 8))
    random_image = data.sample(n=9)
    random_image_paths = random_image['Id'].values
    random_image_cat = random_image['Class'].values

    for index, path in enumerate(random_image_paths):
        im = PIL.Image.open(os.path.join(path_to_images, path))
        plt.subplot(3, 3, index + 1)
        plt.imshow(im)
        plt.title('Class: ' + str(random_image_cat[index]))
        plt.axis('off')
    plt.show()
    return fig


def plot_sample_predict(table, path_to_images):
    plt.figure(figsize=(60, 30))
    data = pd.DataFrame({'Id': table['path'].values,
                         'True class': table['true_label'].values,
                         'Predicted class': table['predict_label'].values})
    random_image = data.sample(n=100)
    random_image_paths = random_image['Id'].values
    random_image_cat = random_image['True class'].values
    random_image_predict = random_image['Predicted class'].values

    for index, path in enumerate(random_image_paths):
        im = PIL.Image.open(os.path.join(path_to_images, path))
        plt.subplot(10, 10, index + 1)
        plt.imshow(im)
        plt.title(f'True class: {str(random_image_cat[index])},\n predicted class: {str(random_image_predict[index])}')
        plt.axis('off')
    plt.show()


def plot_classes_balance(data):
    fig = plt.figure(figsize=(8, 30))
    sns.countplot(data=data, y='Class')
    plt.show()


def plot_sample_images_generator(generator, count_images=6):
    x, y = generator.next()
    fig = plt.figure(figsize=(30, 10))
    for i in range(0, count_images):
        image = x[i]
        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
    plt.show()
    return fig


def plot_history(history, step=0):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'])
    axes[0].plot(history.history['val_accuracy'])
    axes[0].set_title('Model accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Test'])

    axes[1].plot(history.history['loss'])
    axes[1].plot(history.history['val_loss'])
    axes[1].set_title('Model loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Test'])
    return fig
