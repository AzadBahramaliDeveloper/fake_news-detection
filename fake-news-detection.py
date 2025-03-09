import pandas as pd
import numpy as np

true_data = pd.read_csv('/Users/diana/PycharmProjects/fake_news-detection/dataset/true.csv')
fake_data = pd.read_csv('/Users/diana/PycharmProjects/fake_news-detection/dataset/fake.csv')


print("The true data:")
print(true_data.head())
print(true_data.info())
print(true_data.isnull().sum())

print("\nThe fake data:")
print(fake_data.head())
print(fake_data.info())
print(fake_data.isnull().sum())

print("Duplicates in the true dataset:", true_data.duplicated().sum())
print("Duplicates in the fake dataset:", fake_data.duplicated().sum())

clean_true_dataset = true_data.drop_duplicates()
clean_fake_dataset = fake_data.drop_duplicates()
print("Duplicated rows has removed from the True dataset:", clean_true_dataset.duplicated().sum())
print("Duplicated rows has removed from the Fake dataset:", clean_fake_dataset.duplicated().sum())