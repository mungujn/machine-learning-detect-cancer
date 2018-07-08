from functions.transform_images import performTransformationSetWithBlurAndSharpen, \
    resize, getFileAddresses, convertToRGB, toNpArray
import numpy as np
from functions import Timer
timer = Timer()

training_base = "J:\\final year project\\code and models\\data\\augmented\\training\\"

benign_training_folder = training_base + "benign"
malignant_training_folder = training_base + "malignant"

benign_training_image_files = getFileAddresses(benign_training_folder)
malignant_training_image_files = getFileAddresses(malignant_training_folder)

benign_training_images_np_file = benign_training_folder + ".npy"
malignant_training_images_np_file = malignant_training_folder + ".npy"

benign_training_labels_np_file = benign_training_folder + "_labels.npy"
malignant_training_labels_np_file = malignant_training_folder + "_labels.npy"


testing_base = "J:\\final year project\\code and models\\data\\augmented\\testing\\"

benign_testing_folder = testing_base + "benign"
malignant_testing = testing_base + "malignant"

benign_testing_image_files = getFileAddresses(benign_testing_folder)
malignant_testing_image_files = getFileAddresses(malignant_testing)

benign_testing_images_np_file = benign_testing_folder + ".npy"
malignant_testing_images_np_file = malignant_testing + ".npy"

benign_testing_labels_np_file = benign_testing_folder + "_labels.npy"
malignant_testing_labels_np_file = malignant_testing + "_labels.npy"


def transformImages(files):
    for file in files:
        performTransformationSetWithBlurAndSharpen(file)


def convertImagesToRGB(files):
    for file in files:
        convertToRGB(file)


def resizeImages(files):
    for file in files:
        resize(file, (224, 224))


def augmentAndPreprocessTrainingImages():
    transformImages(benign_training_image_files)
    transformImages(malignant_training_image_files)

    convertImagesToRGB(benign_training_image_files)
    convertImagesToRGB(malignant_training_image_files)

    resizeImages(benign_training_image_files)
    resizeImages(malignant_training_image_files)


def preprocessTestingImages():
    convertImagesToRGB(benign_testing_image_files)
    convertImagesToRGB(malignant_testing_image_files)

    resizeImages(benign_testing_image_files)
    resizeImages(malignant_testing_image_files)


def convertImageFilesToNpArray(files):
    first_image = toNpArray(files[0])
    try:
        color_channels = first_image.shape[2]
    except Exception:
        color_channels = 1

    dimension1 = len(files)
    dimension2 = first_image.shape[0]
    dimension3 = first_image.shape[1]

    if color_channels == 3:
        dimension4 = 3
        image_array = np.zeros((dimension1, dimension2, dimension3, dimension4))
    else:
        image_array = np.zeros((dimension1, dimension2, dimension3))

    labels_array = np.zeros((dimension1, 1), dtype=np.int32)
    for i in range(dimension1):
        image_array[i] = toNpArray(files[i])

    if files == benign_training_image_files or files == benign_testing_image_files:
        for i in range(dimension1):
            labels_array[i] = 0
    elif files == malignant_training_image_files or files == malignant_training_image_files:
        for i in range(dimension1):
            labels_array[i] = 1

    if files == benign_training_image_files:
        np.save(benign_training_images_np_file, image_array)
        np.save(benign_training_labels_np_file, labels_array)
    elif files == benign_testing_image_files:
        np.save(benign_testing_images_np_file, image_array)
        np.save(benign_testing_labels_np_file, labels_array)
    elif files == malignant_training_image_files:
        np.save(malignant_training_images_np_file, image_array)
        np.save(malignant_training_labels_np_file, labels_array)
    elif files == malignant_testing_image_files:
        np.save(malignant_testing_images_np_file, image_array)
        np.save(malignant_testing_labels_np_file, labels_array)

    return image_array, labels_array

# helper function for getting training data
def getTrainingDataNp(reload=False):
    if reload:
        x_benign, y_benign = convertImageFilesToNpArray(benign_training_image_files)
        x_malignant, y_malignant = convertImageFilesToNpArray(malignant_training_image_files)

        x = np.concatenate((x_benign, x_malignant))
        y = np.concatenate((y_benign, y_malignant))

        return x, y
    else:
        x_benign = np.load(benign_training_images_np_file)
        y_benign = np.load(benign_testing_labels_np_file)

        x_malignant = np.load(malignant_training_images_np_file)
        y_malignant = np.load(malignant_training_labels_np_file)

        x = np.concatenate((x_benign, x_malignant))
        y = np.concatenate((y_benign, y_malignant))
        return x, y


# helper function for getting testing data
def getTestingDataNp(reload=False):
    if reload:
        x_benign, y_benign = convertImageFilesToNpArray(benign_testing_image_files)
        x_malignant, y_malignant = convertImageFilesToNpArray(malignant_testing_image_files)

        x = np.concatenate((x_benign, x_malignant))
        y = np.concatenate((y_benign, y_malignant))

        return x, y
    else:
        x_benign = np.load(benign_training_images_np_file)
        y_benign = np.load(benign_testing_labels_np_file)

        x_malignant = np.load(malignant_testing_images_np_file)
        y_malignant = np.load(malignant_testing_labels_np_file)

        x = np.concatenate((x_benign, x_malignant))
        y = np.concatenate((y_benign, y_malignant))
        return x, y

#helper function for loading data
def getData():
    timer.start()
    x, y = getTrainingDataNp(reload=True)
    print(x.shape)
    print(y.shape)
    timer.stop("loading training data")

    timer.start()
    x_test, y_test = getTestingDataNp(reload=True)
    print(x_test.shape)
    print(y_test.shape)
    timer.stop("loading testing data")

    return x, y, x_test, y_test


# confirm data save
x_train, y_train, x_test, y_test = getData()
print('Training data shape: ', x_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)
