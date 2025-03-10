
# Dataset Documentation: Fake News Detection

## 1. Overview

This dataset is used for training a machine learning model to classify news articles as either Fake or Real. It consists of textual news content and corresponding labels.

## 2. Data Structure

The dataset contains 9,900 rows and 2 columns:

- Text (object): The content of the news article.

- Label (object): The classification of the article, either "Fake" or "Real".

## 3. Data Cleaning Process

3.1 Handling Missing Values

- No missing values were found in either the "Text" or "Label" columns.

3.2 Identifying and Removing Duplicates

- Initially, 35 duplicate rows were detected.

- All duplicate rows were successfully removed, ensuring that the dataset contains only unique news articles.

4. Dataset Quality Summary

✅ No missing values

✅ No duplicate rows

✅ Balanced structure with well-defined labels

📂 Memory usage: 154.8 KB