import pandas as pd

true_and_fake_data = pd.read_csv('/Users/diana/PycharmProjects/fake_news-detection/dataset/fake_and_real_news.csv')


print("The true data:")
print(true_and_fake_data.head())
print(true_and_fake_data.info())
print(true_and_fake_data.isnull().sum())

print("Duplicates in the dataset:", true_and_fake_data.duplicated().sum())

clean_true_dataset = true_and_fake_data.drop_duplicates()
print("Duplicated rows has removed from the dataset. Updated result:", clean_true_dataset.duplicated().sum())


