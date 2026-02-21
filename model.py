from sklearn.model_selection import train_test_split
import pandas as pd


def load_data():
    data = pd.read_csv('ISIC_2020_Training_GroundTruth.csv')
    train, test = train_test_split(data, test_size=0.2, random_state=42)


    return train, test

if __name__ == "__main__":
    train, test = load_data()
    print(train.head())
    print(test.head())