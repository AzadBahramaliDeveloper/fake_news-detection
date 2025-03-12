import string

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from keras.src.callbacks import history
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


true_and_fake_data = pd.read_csv('/Users/diana/PycharmProjects/fake_news-detection/dataset/fake_and_real_news.csv')

print("The true data:")
print(true_and_fake_data.head())
print(true_and_fake_data.info())
print(true_and_fake_data.isnull().sum())

print("Duplicates in the dataset:", true_and_fake_data.duplicated().sum())

clean_true_dataset = true_and_fake_data.drop_duplicates()
print("Duplicated rows has removed from the dataset. Updated result:", clean_true_dataset.duplicated().sum())

# Download essential resources for text processing using the nltk library
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


# Function to clean and preprocess the text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords and apply stemming
    words = [stemmer.stem(word) for word in words if word not in stop_words]

    # Join words back into a cleaned sentence
    cleaned_text = ' '.join(words)

    return cleaned_text

# Create a copy of the dataset
clean_true_dataset = clean_true_dataset.copy()

# Modify the 'cleaned_text' column using .loc to avoid the SettingWithCopyWarning
clean_true_dataset.loc[:, 'cleaned_text'] = clean_true_dataset['Text'].apply(preprocess_text)

# Check the first few rows of the cleaned text
print(clean_true_dataset[['Text', 'cleaned_text']].head())

# Split the data into training and testing sets
X = clean_true_dataset['cleaned_text']
y = clean_true_dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data size: {len(X_train)}")
print(f"Test data size: {len(X_test)}")

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data, then transform the test data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Training feature shape: {X_train_tfidf.shape}")
print(f"Test feature shape: {X_test_tfidf.shape}")

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Detailed classification report
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# Count of Fake and Real news
plt.figure(figsize=(6, 4))
sns.countplot(data=clean_true_dataset, x="label", palette="viridis")
plt.title("Distribution of Fake vs. Real News")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.show()

# Generate a word cloud for fake news
fake_text = " ".join(clean_true_dataset[clean_true_dataset["Label"] == "Fake"]["cleaned_text"])
wordcloud_fake = WordCloud(width=800, height=400, background_color="black").generate(fake_text)

# Generate a word cloud for real news
real_text = " ".join(clean_true_dataset[clean_true_dataset["Label"] == "Real"]["cleaned_text"])
wordcloud_real = WordCloud(width=800, height=400, background_color="white").generate(real_text)

# Plot the word clouds
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(wordcloud_fake, interpolation="bilinear")
ax[0].set_title("Most Common Words in Fake News")
ax[0].axis("off")

ax[1].imshow(wordcloud_real, interpolation="bilinear")
ax[1].set_title("Most Common Words in Real News")
ax[1].axis("off")

plt.show()

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Epochs")
plt.legend()
plt.show()