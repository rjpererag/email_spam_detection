# Email Classification Project

## Overview

This project focuses on the development of a deep learning model for predicting whether an email is spam or not. The dataset used for this task is obtained from Kaggle and can be found at the following URL: [Email Classification Dataset](https://www.kaggle.com/code/aniketkadam702030/email-classfication-with-tensorflow).

The project is structured to include preprocessing steps for the text data, tokenization using TensorFlow, building and training a neural network model, and finally, testing the model on a separate dataset. Below is a detailed guide on each step:

## 1. Data Preprocessing (preprocess.py)

In the `preprocess.py` file, various text preprocessing techniques are applied to enhance the quality of the input data. These include:

- **Lowercasing:** Convert all text to lowercase.
- **Removing Stop Words:** Eliminate common words that do not contribute much to the meaning.
- **Removing Punctuation:** Get rid of unnecessary punctuation marks.
- **One-Hot Encoding:** Convert the "Spam" or "Not Spam" labels into a binary matrix (0 or 1).

These preprocessing steps aim to clean and standardize the text data, making it suitable for training a deep learning model.

## 2. Tokenization using TensorFlow

Tokenization is a critical step in natural language processing. In this project, the TensorFlow library is employed to tokenize the preprocessed data. The tokenized sequences serve as the input for the subsequent steps in the model-building process.

## 3. Model Architecture

The model architecture consists of an Embedding layer, Global Average Pooling layer, Dense layer with ReLU activation, Dropout layer, and a final Dense layer with a Sigmoid activation function. These layers are designed to process and extract features from the tokenized sequences optimized with "adam" and loss function calculated with "binary_crossentropy". The metric gotten is accuracy with a validation size of 10%.
.

## 4. Model Testing
The trained model is tested using a separate dataset to evaluate its performance and assess its ability to generalize to new data.

## 5. Building a Demo
A demo application can be built to showcase the model's functionality. This could involve taking user input, preprocessing the text, tokenizing it, and using the trained model to predict whether the input corresponds to a spam or non-spam email.