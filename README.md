
# IMDb Reviews Sentiment Analysis

## Overview

This project demonstrates how to perform sentiment analysis on IMDb movie reviews using Python. The goal is to build a machine learning model that can predict the sentiment (positive or negative) of a movie review based on its text.

## Dataset

The dataset used for this sentiment analysis task contains two columns:

1. **review**: This column contains the text of IMDb movie reviews.
2. **sentiment**: This column contains the corresponding sentiment label for each review. The sentiment label is binary, with 0 representing a negative sentiment and 1 representing a positive sentiment.

Please ensure that your dataset is in CSV format with the appropriate column names ('reviews' and 'sentiment').

## Dependencies

Before running the notebook, make sure you have the following Python libraries installed in your environment:

- pandas
- scikit-learn
- nltk

You can install the required libraries using the following command:

```bash
pip install pandas scikit-learn nltk
```

## Notebook Workflow

The sentiment analysis is performed in a Jupyter notebook named "IMDb_Reviews_Sentiment_Analysis.ipynb." The notebook follows these major steps:

1. **Importing Libraries**: The necessary libraries are imported for data manipulation, text preprocessing, and machine learning.

2. **Loading and Preprocessing the Dataset**: The dataset is loaded from the CSV file, and the text data is preprocessed. The preprocessing steps include tokenization, converting text to lowercase, lemmatization, and removing stop words.

3. **Data Splitting**: The dataset is split into training and testing sets. 80% of the data is used for training, and 20% is used for testing the model.

4. **Creating TF-IDF Features**: The text data is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer. This step converts the text into numerical vectors that can be used as input for machine learning models.

5. **Training the Model**: A Naive Bayes classifier (MultinomialNB) is used to train the sentiment analysis model. The model is trained on the training data with corresponding sentiment labels.

6. **Making Predictions and Evaluation**: The trained model is used to predict sentiment labels for the test data. The accuracy, classification report, and confusion matrix are calculated to evaluate the model's performance.
