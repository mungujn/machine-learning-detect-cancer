# Classification of Lesions in Breast Ultrasound Umages Using Neural Networks

This repo contains source code for my undergraduate dissertation from Makerere University College of Engineering, Design, Art and Technology.

The goal of this project was to build and evaluate a neural network framework for classification of lesions in breast ultrasound images. A classification model based on k-Nearest Neighbour (k-NN) algorithm was built to serve as an evaluation baseline. 4 neural network models were then built using the TensorFlow and Keras deep learning libraries; A fully connected neural network, a custom Convolutional Neural Network (CNN) and two transfer learning networks based on retraining InceptionV3 which is a state of the art general purpose image classification CNN. Neural network approaches outperformed the k-NN. The CNN manages to achieve a low false negative rate and high true positive rate hence a high sensitivity of 0.85. The transfer learning networks underperform due to data limitations.

There are 8 main jupyter notebooks in this project

* 1 data.ipynb

   Contains a brief description of the clinical data used and how it was pre-processed it
  * functions/transfor_images.py

      A "manual" implementation of image pre processing. Both keras and tensorflow have good image pre-processing cabalities that can be used.
* 2 knn.ipynb

   Using the k-NN algorithm for lesion classification
* 3 fully connected net

   Building a fully connected nueral network for lesion classification
* 4.1 CNN evaluating learning

   Evaluating different learning algorithms
* 4.2 CNN best

   The best performing CNN
* 4.3 CNN with longer epochs.ipynb and 4.4 CNN unbalanced data, longer epochs.ipynb

   Further CNN experimentation

* 5 fine tuning inceptionv3 for bus

   Using transfer learning for lesion classification

I also built a web application (React.js based) to demonstrate a potential application and deployment scenario for the models that were built during this project.

1. A radiologist or sonographer would log into the application

![Sign in screen for the app](https://github.com/mungujn/machine-learning-detect-cancer/raw/master/images/login.PNG "Sign in screen for the app")
![After signing in and before uploading a scan](https://github.com/mungujn/machine-learning-detect-cancer/raw/master/images/after_sign_in.PNG "After signing in and before uploading a scan")

1. After signing into the application the user uploads a breast ultrasound scan for automated analysis

![After uploading an Ultrasound scan the system performs analysis](https://github.com/mungujn/machine-learning-detect-cancer/raw/master/images/processing.PNG "After uploading an Ultrasound scan the system performs analysis")
![After analysis using the CNN model a lesion classification is displayed](https://github.com/mungujn/machine-learning-detect-cancer/raw/master/images/analysis_complete.PNG "After analysis using the CNN model a lesion classification is displayed")

For a more detailed description of the project, take a look at my full dissertation PDF file in this repo