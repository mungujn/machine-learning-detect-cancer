import numpy as np
from functions.timing import Timer
timer = Timer()
training_base = "J:\\final year project\\code and models\\data\\augmented\\training\\"

benign_training_folder = training_base + "benign"
malignant_training_folder = training_base + "malignant"


benign_training_images_np_file = benign_training_folder + "_training.npy"
malignant_training_images_np_file = malignant_training_folder + "_training.npy"

benign_training_labels_np_file = benign_training_folder + "_training_labels.npy"
malignant_training_labels_np_file = malignant_training_folder + "_training_labels.npy"


testing_base = "J:\\final year project\\code and models\\data\\original\\testing\\"

benign_testing_folder = testing_base + "benign"
malignant_testing_folder = testing_base + "malignant"

benign_testing_images_np_file = benign_testing_folder + "_testing.npy"
malignant_testing_images_np_file = malignant_testing_folder + "_testing.npy"

benign_testing_labels_np_file = benign_testing_folder + "_testing_labels.npy"
malignant_testing_labels_np_file = malignant_testing_folder + "_testing_labels.npy"


def getTrainingDataNp():
    # timer.start()
    x_benign = np.load(benign_training_images_np_file)
    y_benign = np.load(benign_training_labels_np_file)

    x_malignant = np.load(malignant_training_images_np_file)
    y_malignant = np.load(malignant_training_labels_np_file)

    x = np.concatenate((x_benign, x_malignant))
    y = np.concatenate((y_benign, y_malignant))

    # timer.stop("getting training data")
    return x, y


def getTestingDataNp():
    # timer.start()
    x_benign = np.load(benign_testing_images_np_file)
    y_benign = np.load(benign_testing_labels_np_file)

    x_malignant = np.load(malignant_testing_images_np_file)
    y_malignant = np.load(malignant_testing_labels_np_file)

    x = np.concatenate((x_benign, x_malignant))
    y = np.concatenate((y_benign, y_malignant))

    # timer.stop("getting testing data")
    return x, y


def getData():
    x, y = getTrainingDataNp()
    print(x.shape)
    print(y.shape)

    x_test, y_test = getTestingDataNp()
    print(x_test.shape)
    print(y_test.shape)

    return x, y, x_test, y_test

def getTestData():
    x_test, y_test = getTestingDataNp()
    # print(x_test.shape)
    # print(y_test.shape)

    return x_test, y_test