# Fake News Detection Project

## Problem Statement

The goal of this project is to build a machine learning model that can identify fake news articles. We will use two datasets, one containing real news and another containing fake news, to train and evaluate the model.

## Dataset Overview

Two CSV files are used in this project:

- Real News Dataset (21,417 entries)

- Fake News Dataset (23,481 entries)

## Dataset Structure

Both datasets contain the following columns:

- title - The headline of the news article.

- text - The main content of the article.

- subject - The category of the news.

- date - The publication date.

## Initial Data Analysis

**Missing Values Check:** 

- No missing values found in either dataset.

- Command used: df.isnull().sum() ✅

**Data Types:**

- All columns are of type object (text).

- The date column may require preprocessing.

**Duplicate Check:**

- Running df.duplicated().sum() to identify duplicate rows.

- Need to decide whether to remove duplicates if found.

# Next Steps

1. Handle duplicate rows (if necessary).

2. Preprocess the data (e.g., convert dates, clean text).

3. Split data into training and testing sets.

4. Train a machine learning model.

5. Evaluate and optimize the model.

6. Deploy the model.

# Data Cleaning: Handling Duplicates

**Dataset Overview:**
- The two datasets (True and Fake) were initially inspected for duplicates. The True dataset contained 206 duplicate rows, and the Fake dataset contained 3 duplicate rows.

**Data Cleaning Process:**
- After detecting duplicates, both datasets were cleaned by removing duplicate rows using the drop_duplicates() method in pandas.

**Outcome:**
- After cleaning, no duplicate rows remain in either dataset, confirming that the data is now free of duplicates.

# Repository Setup

**GitHub Repository Created: ✅**

**Files Pushed to GitHub: ✅**
