
# IMDb Reviews Sentiment Analysis

## Overview

This project demonstrates how to perform sentiment analysis on IMDb movie reviews using Python. The goal is to build a machine learning model that can predict the sentiment (positive or negative) of a movie review based on its text.

## Dataset

The dataset used for this sentiment analysis task contains two columns:

1. **reviews**: This column contains the text of IMDb movie reviews.
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

## How to Use

1. Make sure you have installed the required dependencies as mentioned in the "Dependencies" section.

2. Prepare your dataset in CSV format with two columns: 'reviews' and 'sentiment'.

3. Replace `'path_to_your_dataset.csv'` in the notebook with the actual path to your dataset file.

4. Run each cell in the notebook to perform data preprocessing, model training, and evaluation.

5. After running the entire notebook, you will see the accuracy, classification report, and confusion matrix, which will provide insights into the model's performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project was obtained from IMDb (Internet Movie Database) and is not the property of the author. The dataset is for educational and research purposes only.

## Disclaimer

This project is for educational purposes only and is not intended for any commercial use. The author does not claim ownership of the dataset used for training the model and is not responsible for any misuse of the code or the dataset.

---
Replace `[LICENSE]` with the appropriate path to the license file, if applicable. Also, feel free to modify the "Acknowledgments" section to credit the sources of the dataset, if any.

Save this content in a file named `README.md`, and it will serve as the documentation for your IMDb reviews sentiment analysis project.
