import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
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

