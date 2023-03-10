# Quora Kaggle Dataset Competition

This repository contains my solution for the IN4325 Information Retrieval challenge. My strategy consists of:

## Normalization

- Convert the questions to lowercase.
- Remove non-alphanumeric characters.
- Filter the stop words using nltk's stopwords list.
- Lemmatizing the questions.

## Feature engineering

- Calculate the Levenshtein distance.
- Calculate the cosine distance using sent2vec.
- Calculate the number of common words.

## Classification

Split the data 80/20 to train the following classifiers and then classify the test set:

- RandomForestClassifier
- DecisionTreeClassifier
- KNeighborsClassifier
- LogisticRegression
