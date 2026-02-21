from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd


def load_data():
    data = pd.read_csv('ISIC_2020_Training_GroundTruth.csv')
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

def train_model(train):
    X_train = train.drop(columns=['target', 'benign_malignant'])
    y_train = train['target']
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    train, test = load_data()
    model = train_model(train)
    