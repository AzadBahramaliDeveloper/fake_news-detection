import string
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Read data
true_and_fake_data = pd.read_csv('/Users/diana/PycharmProjects/fake_news-detection/dataset/fake_and_real_news.csv')

# Clean data
clean_true_dataset = true_and_fake_data.drop_duplicates()
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    cleaned_text = ' '.join(words)
    return cleaned_text

# Explicitly create a copy of the DataFrame
clean_true_dataset = true_and_fake_data.drop_duplicates().copy()

# Then apply the transformation
clean_true_dataset.loc[:, 'cleaned_text'] = clean_true_dataset['Text'].apply(preprocess_text)


# Split data
X = clean_true_dataset['cleaned_text']
y = clean_true_dataset['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# Visualizations
plt.figure(figsize=(6, 4))
sns.countplot(data=clean_true_dataset, x="label", hue="label", palette="viridis", legend=False)
plt.title("Distribution of Fake vs. Real News")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.show()

# Word Cloud for Fake News
fake_text = " ".join(clean_true_dataset[clean_true_dataset["label"] == "Fake"]["cleaned_text"])
wordcloud_fake = WordCloud(width=800, height=400, background_color="black").generate(fake_text)

# Word Cloud for Real News
real_text = " ".join(clean_true_dataset[clean_true_dataset["label"] == "Real"]["cleaned_text"])
wordcloud_real = WordCloud(width=800, height=400, background_color="white").generate(real_text)

# Plot word clouds
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(wordcloud_fake, interpolation="bilinear")
ax[0].set_title("Most Common Words in Fake News")
ax[0].axis("off")

ax[1].imshow(wordcloud_real, interpolation="bilinear")
ax[1].set_title("Most Common Words in Real News")
ax[1].axis("off")

plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
