# Approach

## Method 1: Support vector classifier, Naive Bayes and Random Forrest Classifier using Sklearn library
The pipeline consists of three main steps:
1. Vectorizing the text based on TF-IDF scores (includes removal of stop words)
2. Fit the model and validate for train data
3. Obtain predictions for the test data using the fitted model


## Method 2: CNN classifier using PyTorch
The pipeline consists of the following steps:
1. Build the vocabulary using train data and vectorize the data using the vocabulary
2. Train the CNN model using the train data for 'n' epochs and validate on the test data
3. Obtain predictions for the test data using the trained model

## Evaluation and train-test split
For both the methods train and validation split ratio was 0.8 and 0.2 respectively

Evaluation measures used are accuracy and F1 score

Observations:
1. With method 1, accuracy was on average 53-55 %, F1 score: 55 %
2. With method 2, accuracy was on average 65-68 %, F1 score: 64 %

## Inference:
The accuracy in general for both methods is less owing to the fact that the number of training samples used is less. As well as no pretrained model is used to make up for less training data. The performace of method 1 is less than in method 2 because of lesser capacity of the model to capture the complexity of th data.




