import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score

def load_data():
    # 1. Load the dataset
    data = pd.read_csv('train.csv')
    
    # 2. Select predictive features and target
    # We use 'sex', 'age_approx', and 'anatom_site_general_challenge'
    features = ['sex', 'age_approx', 'anatom_site_general_challenge', 'width','height']
    target = 'target'
    
    X = data[features]
    y = data[target]
    
    # 3. Split the data (80% train, 20% test)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_model():
    """ Conducts prepossing to handle missing values and categorical variables, then builds a Logistic Regression model. """
    numeric_features = ['age_approx', 'width', 'height']
    categorical_features = ['sex', 'anatom_site_general_challenge']

   
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

   
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, penalty='l2', C=1.0, solver='lbfgs', random_state=42))
    ])
    
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    
    
    model = build_model()
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, y_pred))
    # confusion_matrix = confusion_matrix(y_test, y_pred)

