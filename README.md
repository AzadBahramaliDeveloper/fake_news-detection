
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

### Problem Definition
The task is to classify news articles as either "Fake" or "Real" using machine learning. The problem involves training a model that can accurately distinguish between fake and real news articles based on their textual content.

### Dataset
The dataset used in this project is a CSV file containing two primary columns:

### Text: The content of the news article.
Label: A label indicating whether the article is "Fake" or "Real".
The dataset was initially cleaned by removing duplicate entries.

Data Preprocessing
The following preprocessing steps were applied to the dataset:

Text Lowercasing: All text was converted to lowercase to standardize the input.
Punctuation Removal: All punctuation marks were removed from the text to focus only on the words.
Tokenization: The text was split into individual words (tokens).
Stopwords Removal: Common words (such as "and", "the", "is") were removed to improve model efficiency.
Stemming: Words were reduced to their root form using the Porter Stemmer, e.g., "running" becomes "run".

### Model Selection
A Logistic Regression model was chosen for the classification task. This model was trained using TF-IDF (Term Frequency - Inverse Document Frequency) vectorization, which transforms the text data into numerical features based on the importance of words in each article.

### Training and Evaluation
Train-Test Split: The data was split into training (80%) and testing (20%) sets to evaluate model performance.
Metrics: The model’s accuracy was calculated on the test set, and a classification report was generated. Additionally, a confusion matrix was used to visualize misclassifications between fake and real news.

### Visualizations
Distribution of Fake vs. Real News: A count plot was generated to visualize the distribution of fake and real news articles in the dataset.

Word Clouds: Word clouds were created for both fake and real news articles to identify the most common words within each category.

The word cloud for fake news highlights words typically associated with misleading or exaggerated content.
The word cloud for real news displays more neutral or factual language.

### Confusion Matrix
A confusion matrix was plotted to show the true positive, true negative, false positive, and false negative values of the model's predictions. This visualization helps assess the model's ability to distinguish between fake and real news.


### Dataset resources
- dataset : https://www.kaggle.com/datasets/vishakhdapat/fake-news-detection